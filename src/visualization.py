import open3d as o3d
import torch


def create_lines(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    num_points_per_line: int,
):
    """
    rays_origins : N rays x 3
    rays_directions : N rays x 3
    t_start : N_rays x 1
    t_start : N_rays x 1
    num_points_per_line: int

    # Open3d lines are too thin and cannot control the visuals easily. Use point cloud instead
    """
    dists = (t_end - t_start) / num_points_per_line  # N_rays x 1

    ts = (
        t_start + dists * torch.arange(num_points_per_line).to(ray_origins)
    ).unsqueeze(
        -1
    )  # N_rays x num_points_per_line x 1

    ray_offsets = (
        ray_directions.view(-1, 1, 3) * ts
    )  # N_rays x 1 x 3  and N_rays x num_points_per_line x 1 -> N_rays x num_points_per_line x 3

    # N_rays x 1 x 3 and N_rays x num_points_per_line x 3
    points = ray_origins.view(-1, 1, 3) + ray_offsets
    return o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points.view(-1, 3).detach().cpu().numpy())
    )
