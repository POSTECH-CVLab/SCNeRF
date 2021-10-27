import numpy as np
import imageio
import os

import torch


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    return imgs


def load_custom_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, args=None):
    

    imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    
    print('Data:')

    if args.llffhold > 0:
        i_test = np.arange(images.shape[0])[::args.llffhold]

    print('HOLDOUT view is', i_test)

    poses = np.zeros((len(images), 3, 5)).astype(np.float32)

    i_train = np.array([i for i in range(len(poses)) if not i in i_test])

    poses[:, :3, :3] = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])[None]
    poses[:, :3, 3] = 0.
        
    H, W = imgs[0].shape[:2]
    intrinsic_gt = torch.tensor([
        [W, 0, W//2, 0],
        [0, H, H//2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).cuda()
    poses[:, 0, 4]  = H
    poses[:, 1, 4]  = W
    poses[:, 2, 4]  = H
    
    images = images.astype(np.float32)

    extrinsics_gt = torch.zeros((len(poses), 4, 4)).cuda()
    extrinsics_gt[:, :3, :4] = torch.from_numpy(poses[:, :, :4]).cuda()
    extrinsics_gt[:, 3, 3] = 1

    return images, poses, None, None, i_test, \
        (intrinsic_gt.float(), extrinsics_gt.float())