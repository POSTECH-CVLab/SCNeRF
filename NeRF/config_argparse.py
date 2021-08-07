import configargparse

import sys
sys.path.insert(0, "..")
from src.utils import str2bool


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_per_gpu", type=int, default=1024 * 64 * 4,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_iters", type=int, default=None,
                        help='number of iterations')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    # Ray Distance Loss Settings
    parser.add_argument(
        "--ray_loss_type",
        type=str,
        choices=[
            "none",
            "proj_ray_dist"
        ],
        default="none",
        help="ray distance loss type",
    )

    # :wq Settings
    parser.add_argument(
        "--superglue_weight",
        type=str,
        default="outdoor",
        choices=[
            "indoor",
            "outdoor"
        ],
        help="Use outdoor weight or indoor weight",
    )
    parser.add_argument(
        "--max_keypoints", type=int, default=1024, help="Max number of keypoints"
    )
    parser.add_argument(
        "--nms_radius", type=int, default=4, help="NMS radius for SuperGlue"
    )
    parser.add_argument(
        "--sinkhorn_iterations",
        type=int,
        default=20,
        help="Number of sinkhorn iteration",
    )
    parser.add_argument(
        "--match_threshold", type=float, default=0.2, help="Match threshold"
    )
    parser.add_argument(
        "--keypoint_threshold", type=float, default=0.005, help="Threshold of keypoints"
    )
    parser.add_argument(
        "--match_num",
        type=int,
        default=50,
        help="Number of match will be used in ray distance loss",
    )
    parser.add_argument(
        "--multiplicative_noise",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="increase noise proportional to target camera parameter",
    )

    # logger
    parser.add_argument(
        "--logger",
        type=str,
        choices=[
            "wandb"
        ],
        default="wandb",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Debug code",
    )

    parser.add_argument(
        "--pairing_angle_threshold",
        type=float,
        default=30,
        help=\
        """
        Generate image pairs only if the absolue angle difference between
        two images is below the threshold. Input is a degree form.
        """
        )

    # Injecting noises in camera parameters
    parser.add_argument(
        "--initial_noise_size_intrinsic",
        type=float,
        default=0.00,
        help="initial noise size in intrinsic parameters"
    )

    parser.add_argument(
        "--initial_noise_size_translation",
        type = float,
        default = 0.00,
        help = "initial noise size in translation paraemters",
    )

    parser.add_argument(
        "--initial_noise_size_rotation",
        type = float,
        default = 0.00,
        help = "initial noise size in rotation matrices",
    )

    parser.add_argument(
        "--camera_model",
        type=str,
        choices=[
            "none",
            "pinhole_rot_noise",
            "pinhole_rot_noise_dist",
            "pinhole_rot_noise_extrinsics_only",
            "pinhole_rot_noise_10k_rayo_rayd",
            "pinhole_rot_noise_no_multi_on_trans", 
            "pinhole_rot_noise_10k_rayo_rayd_dist"
        ],
        default="none",
        help="camera model in use",
    )

    parser.add_argument(
        "--non_linear_weight_decay", 
        type=float, 
        default=0.0, 
        help="weight decay on non-linear distortion"
    )
    parser.add_argument(
        "--i_ray_dist_loss", 
        type=int, 
        default=10, 
        help="Alternating frequency"
    )
    parser.add_argument(
        "--ray_dist_loss_weight",
        type=float,
        default=1.0,
        help="ratio between ray distance loss and photometric loss"
    )

    parser.add_argument(
        "--proj_ray_dist_threshold",
        default=5.0,
        type=float,
        help="Threshold to filter in projection ray distance loss"
    )

    parser.add_argument(
        "--extrinsics_noise_scale",
        type=float,
        default=1,
        help="extrinisic noise scale when the learning is noise learning. ",
    )
    parser.add_argument(
        "--intrinsics_noise_scale",
        type=float,
        default=1,
        help="intrinisic noise scale when the learning is noise learning.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        help="seed to fix"
    )
    
    parser.add_argument(
        "--run_without_colmap",
        choices=["both", "rot", "trans", "none"],
        default="none",
        help= """
        Run without colmap setup
        rot: rotation matrices are set to the identity matrix.
        trans: translation vectors are set to the zero vector.
        both: rot + trans
        """
    )

    # 10k model parameters
    parser.add_argument("--grid_size", default=10, type=int)
    parser.add_argument("--ray_d_noise_scale", default=1e-4, type=float)
    parser.add_argument("--ray_o_noise_scale", default=1e-4, type=float)

    # Matcher Experiments
    parser.add_argument(
        "--matcher", choices=["superglue", "sift"], default="superglue", 
        type=str
    )
    
    parser.add_argument(
        "--use_custom_optim",
        type=str2bool, 
        nargs="?", 
        const=True,
        default=False, 
        help= "Adopt custom optimizer"
    )

    # Curriculum Learning
    parser.add_argument(
        "--add_ie", default=0, type=int, 
        help="step to start learning ie"
    )
    parser.add_argument(
        "--add_od", default=0, type=int,
        help="step to start learning od"
    )
    parser.add_argument(
        "--add_prd", type=int, default=50000, 
        help="step to use prd loss"
    )


    return parser


