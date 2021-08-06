
import sys
sys.path.append("..")

from src.utils import str2bool
import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    # dataset options
    parser.add_argument("--datadir", type=str, default=None, help='input data directory')
    parser.add_argument("--scene", type=str, default=None, help='scene name')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    # model size
    parser.add_argument("--netdepth", type=int, default=8, help='layers in coarse network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer in coarse network')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    # checkpoints
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    # batch size
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 2,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk_size", type=int, default=1024 * 8,
                        help='number of rays processed in parallel, decrease if running out of memory')
    # iterations
    parser.add_argument("--N_iters", type=int, default=250001,
                        help='number of iterations')
    # render only
    parser.add_argument("--render_splits", type=str, default='test',
                        help='splits to render')
    # cascade training
    parser.add_argument("--cascade_level", type=int, default=2,
                        help='number of cascade levels')
    parser.add_argument("--cascade_samples", type=str, default='64,64',
                        help='samples at each level')
    # multiprocess learning
    parser.add_argument("--world_size", type=int, default='-1',
                        help='number of processes')
    # optimize autoexposure
    parser.add_argument("--optim_autoexpo", action='store_true',
                        help='optimize autoexposure parameters')
    parser.add_argument("--lambda_autoexpo", type=float, default=1., help='regularization weight for autoexposure')

    # learning rate options
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.1,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=750,
                        help='decay learning rate by a factor every specified number of steps')
    # rendering options
    parser.add_argument("--det", action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument("--max_freq_log2", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--max_freq_log2_viewdirs", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--load_min_depth", action='store_true', help='whether to load min depth')
    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    parser.add_argument("--ray_o_noise_scale", type=float, default=1e-3, help="scale of offset distortion")
    parser.add_argument("--ray_d_noise_scale", type=float, default=1e-3, help="scale of direction distortion")
    parser.add_argument("--grid_size", type=int, default=10)

    parser.add_argument("--pairing_angle_threshold", type=float, default=30, help="max pair angle threshold")

    parser.add_argument("--ray_dist_loss_weight", type=float, default=1e-4, help="weight of prd loss")
    parser.add_argument("--alternate_frequency", type=int, default=1, help="frequency of our optimization")

    parser.add_argument("--camera_model", default="pinhole_rot_noise_10k_rayo_rayd", type=str)
    parser.add_argument("--extrinsics_noise_scale", default=1e-2, type=float)
    parser.add_argument("--intrinsics_noise_scale", default=1.0, type=float)
    parser.add_argument("--distortion_noise_scale", default=1e-2, type=float)

    parser.add_argument("--prd_only", action="store_true")

    parser.add_argument("--master_addr", type=int, default=12345)

    # SuperGlue
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

    parser.add_argument("--proj_ray_dist_threshold", type=float, default=5.0)

    parser.add_argument("--camera_log", type=int, default=20000)
    parser.add_argument("--use_custom_optim", action="store_true")
    parser.add_argument("--non_linear_weight_decay", default=0.1)

    parser.add_argument(
        "--run_fisheye", type=str2bool, nargs="?", const=True, 
        default=False, help="run fish-eye specific code"
    )
    parser.add_argument(
        "--use_camera", type=str2bool, nargs="?", const=True, 
        default=False, help="train camera parameters"
    )
    parser.add_argument(
        "--load_camera", type=str2bool, nargs="?", const=True, 
        default=False, help="load the camera parameters"
    )
    parser.add_argument(
        "--load_test", type=str2bool, nargs="?", const=True, 
        default=False, help="load whole parameters when testing"
    )

    parser.add_argument(
        "--add_ie", type=int, default=-1, 
        help="Step to start learning intrinsic and extrinsic parameters"
    )
    parser.add_argument(
        "--add_radial", type=int, default=-1, 
        help="Step to start learning radial distortion parameters"
    )
    parser.add_argument(
        "--add_od", type=int, default=-1, 
        help="Step to start learning ray_o and ray_d parameters"
    )
    parser.add_argument(
        "--add_prd", type=int, default=-1, 
        help="Step to start using prd loss"
    )
    parser.add_argument(
        "--multiplicative_noise", type=str2bool, nargs="?", const=True, 
        default=False, help="learn multiplicative noise"
    )
    parser.add_argument(
        "--normalize_factor", type=float, default=1.0, 
        help="factor to reduce or enlarge coverage of unit sphere"
    )

    return parser