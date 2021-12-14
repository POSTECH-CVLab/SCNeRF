import numpy as np
import os
import imageio
import skimage.transform
import argparse
from subprocess import check_output
import colmap_utils.read_sparse_model as read_model


def load_colmap_data(realdir):

    camerasfile = os.path.join(realdir, "sparse/0/cameras.bin")
    camdata = read_model.read_cameras_binary(camerasfile)

    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print("Cameras", len(camdata.keys()))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h, w, f]).reshape([3, 1])

    imagesfile = os.path.join(realdir, "sparse/0/images.bin")
    imdata = read_model.read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print("Images #", len(names))
    perm = []
    with open(os.path.join(realdir, "train.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            for i in range(len(names)):
                if names[i] == line.strip():
                    perm.append(i)
    if os.path.exists(os.path.join(realdir, "test.txt")):
        with open(os.path.join(realdir, "test.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                for i in range(len(names)):
                    if names[i] == line.strip():
                        perm.append(i)
    if len(names) != len(perm):
        print("COLMAP fails for some images!")
        exit()
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate(
        [poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1
    )

    points3dfile = os.path.join(realdir, "sparse/0/points3D.bin")
    pts3d = read_model.read_points3d_binary(points3dfile)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate(
        [
            poses[:, 1:2, :],
            poses[:, 0:1, :],
            -poses[:, 2:3, :],
            poses[:, 3:4, :],
            poses[:, 4:5, :],
        ],
        1,
    )

    return poses, pts3d, perm


def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print(
                    "ERROR: the correct camera poses for current points cannot be accessed"
                )
                return
            cams[ind - 1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print("Points", pts_arr.shape, "Visibility", vis_arr.shape)

    zvals = np.sum(
        -(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :])
        * poses[:3, 2:3, :],
        0,
    )
    valid_z = zvals[vis_arr == 1]
    print("Depth stats", valid_z.min(), valid_z.max(), valid_z.mean())

    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]
        close_depth, inf_depth = np.percentile(zs, 0.1), np.percentile(zs, 99.9)

        save_arr.append(
            np.concatenate(
                [poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0
            )
        )
    save_arr = np.array(save_arr)

    np.save(os.path.join(basedir, "poses_bounds.npy"), save_arr)


def minify_v0(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, "images_{}".format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, "images_{}x{}".format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    def downsample(imgs, f):
        sh = list(imgs.shape)
        sh = sh[:-3] + [sh[-3] // f, f, sh[-2] // f, f, sh[-1]]
        imgs = np.reshape(imgs, sh)
        imgs = np.mean(imgs, (-2, -4))
        return imgs

    imgdir = os.path.join(basedir, "images")
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [
        f
        for f in imgs
        if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
    ]
    imgs = np.stack([imageio.imread(img) / 255.0 for img in imgs], 0)

    for r in factors + resolutions:
        if isinstance(r, int):
            name = "images_{}".format(r)
        else:
            name = "images_{}x{}".format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        print("Minifying", r, basedir)

        if isinstance(r, int):
            imgs_down = downsample(imgs, r)
        else:
            imgs_down = skimage.transform.resize(
                imgs,
                [imgs.shape[0], r[0], r[1], imgs.shape[-1]],
                order=1,
                mode="constant",
                cval=0,
                clip=True,
                preserve_range=False,
                anti_aliasing=True,
                anti_aliasing_sigma=None,
            )

        os.makedirs(imgdir)
        for i in range(imgs_down.shape[0]):
            imageio.imwrite(
                os.path.join(imgdir, "image{:03d}.png".format(i)),
                (255 * imgs_down[i]).astype(np.uint8),
            )


def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, "images_{}".format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, "images_{}x{}".format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    imgdir = os.path.join(basedir, "images")
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [
        f
        for f in imgs
        if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
    ]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = "images_{}".format(r)
            resizearg = "{}%".format(int(100.0 / r))
        else:
            name = "images_{}x{}".format(r[1], r[0])
            resizearg = "{}x{}".format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print("Minifying", r, basedir)

        os.makedirs(imgdir)
        check_output("cp {}/* {}".format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split(".")[-1]
        args = " ".join(
            ["mogrify", "-resize", resizearg, "-format", "png", "*.{}".format(ext)]
        )
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != "png":
            check_output("rm {}/*.{}".format(imgdir, ext), shell=True)
            print("Removed duplicates")
        print("Done")


def gen_poses(basedir, factors=None):

    print("Post-colmap")

    poses, pts3d, perm = load_colmap_data(basedir)

    save_poses(basedir, poses, pts3d, perm)

    if factors is not None:
        print("Factors:", factors)
        minify(basedir, factors)

    print("Done with imgs2poses")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working_dir", required=True, type=str, help="Path to generate pose files"
    )
    args = parser.parse_args()
    gen_poses(args.working_dir)
