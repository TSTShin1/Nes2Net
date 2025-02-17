# Example for testing WavLM_Nes2Net_X_SeLU
CUDA_VISIBLE_DEVICES=0 python eval.py --base_dir /home/tianchi/data/SVDD2024/test_set \
--model_path "/data/tianchi/Nes2Net_SVDD_ckpts/WavLM_Nes2Net_X_SeLU_e74_seed420_valid0.04245662278274772.pt" \
--agg SEA --pool_func 'mean' --dilation 1 --Nes_ratio 8 8 --SE_ratio 1 --model_name WavLM_Nes2Net_X_SeLU \
--outputname E74_WavLM_SEA_Nes2Net_X_SeLU_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed420

# Example for testing WavLM_Nes2Net_X
CUDA_VISIBLE_DEVICES=0 python eval.py --base_dir /home/tianchi/data/SVDD2024/test_set \
--model_path "/data/tianchi/Nes2Net_SVDD_ckpts/WavLM_Nes2Net_X_e75_seed420_valid0.03192785031473534.pt" \
--agg SEA --pool_func 'mean' --dilation 1 --Nes_ratio 8 8 --SE_ratio 1 --model_name WavLM_Nes2Net_X \
--outputname E75_WavLM_SEA_Nes2Net_X_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed420
# Example for testing WavLM_Nes2Net_X
CUDA_VISIBLE_DEVICES=0 python eval.py --base_dir /home/tianchi/data/SVDD2024/test_set \
--model_path "/home/tianchi/Nes2Net_SVDD/WavLM_Nes2Net_X_e54_seed42_valid0.03344672367129024.pt" \
--agg SEA --pool_func 'mean' --dilation 1 --Nes_ratio 8 8 --SE_ratio 1 --model_name WavLM_Nes2Net_X \
--outputname E54_WavLM_SEA_Nes2Net_X_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed42

# Example for testing WavLM_Nes2Net
CUDA_VISIBLE_DEVICES=0 python eval.py --base_dir /home/tianchi/data/SVDD2024/test_set \
--model_path "/data/tianchi/Nes2Net_SVDD_ckpts/WavLM_Nes2Net_e54_seed42_valid0.03359051995564846.pt" \
--agg SEA --pool_func 'mean' --dilation 1 --Nes_ratio 8 8 --SE_ratio 1 --model_name WavLM_Nes2Net \
--outputname E54_WavLM_SEA_Nes2Net_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed42
