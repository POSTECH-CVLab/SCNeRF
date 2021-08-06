python3 ddp_train_nerf.py \
    --config configs/tanks_and_temples/tat_intermediate_M60_ours.txt \
    --calibrate_camera \
    --use_custom_optim \
    --expname ours_wo_colmap_m60 \
    --without_colmap \
    --extrinsics_noise_scale 1.0 \
    --calibrate_from 50000 \
    --master_addr 12350 
    