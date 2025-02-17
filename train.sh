# Example for training WavLM_Nes2Net
python train.py --base_dir /home/tianchi/data/SVDD2024/ --algo 8 --gpu_id 1 --T_max 5 --epochs 75 --lr 0.000001 --batch_size 34 \
--agg SEA --pool_func 'mean' --dilation 1 --Nes_ratio 8 8 --SE_ratio 1 --model_name WavLM_Nes2Net --seed 42 \
--foldername WavLM_SEA_Nes2Net_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed42

# Example for training WavLM_Nes2Net_X
python train.py --base_dir /home/tianchi/data/SVDD2024/ --algo 8 --gpu_id 2 --T_max 5 --epochs 75 --lr 0.000001 --batch_size 34 \
--agg SEA --pool_func 'mean' --dilation 1 --Nes_ratio 8 8 --SE_ratio 1 --model_name WavLM_Nes2Net_X --seed 420 \
--foldername WavLM_SEA_Nes2Net_X_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed420

# Example for training WavLM_Nes2Net_X_SeLU
python train.py --base_dir /home/tianchi/data/SVDD2024/ --algo 8 --gpu_id 3 --T_max 5 --epochs 75 --lr 0.000001 --batch_size 34 \
--agg SEA --pool_func 'mean' --dilation 1 --Nes_ratio 8 8 --SE_ratio 1 --model_name WavLM_Nes2Net_X_SeLU --seed 420 \
--foldername WavLM_SEA_Nes2Net_X_SeLU_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed420
