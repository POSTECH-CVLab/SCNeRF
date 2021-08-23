cd NeRF

python run_nerf.py \
    --config configs/llff_data/orchids.txt \
    --expname $(basename "${0%.*}") \
    --chunk 8192 \
    --N_rand 1024 \
    --run_without_colmap both \
    --N_iters 800001 \
    --lrate_decay 400
