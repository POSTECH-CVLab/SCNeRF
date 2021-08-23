cd NeRF

python run_nerf.py \
    --config configs/llff_data/leaves.txt \
    --expname $(basename "${0%.*}") \
    --chunk 8192 \
    --N_rand 1024 \
    --camera_model pinhole_rot_noise_10k_rayo_rayd \
    --ray_loss_type proj_ray_dist \
    --multiplicative_noise True \
    --i_ray_dist_loss 10 \
    --grid_size 10 \
    --run_without_colmap both \
    --ray_dist_loss_weight 0.0001 \
    --N_iters 800001 \
    --ray_o_noise_scale 1e-3 \
    --ray_d_noise_scale 1e-3 \
    --add_ie 0 \
    --add_od 200000 \
    --add_prd 400000 \
    --lrate_decay 400 