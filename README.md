# Nes2Net: A Lightweight Nested Res2Net Architecture for Foundation Model-based Speech Anti-spoofing 
Official release of pretrained models and scripts for "Nes2Net: A Lightweight Nested Res2Net Architecture for Foundation Model-based Speech Anti-spoofing."

**This repo is for the Controlled Singing Voice Deepfake Detection (CtrSVDD) dataset.**

For the **ASVspoof** and **In-the-Wild** dataset: Coming soon

For the **PartialSpoof** dataset: Coming soon

arXiv Link: To add

# Update:


# Pretrained Models
| Model                | Pre-trained Checkpoints | Score File       | Seed | Best Valid Epoch | w/o ACE. B.F. | w/ ACE. B.F. |
|----------------------|-------------------------|-------------------|------|------------------|---------------|--------------|
| WavLM_Nes2Net        | -                       | -                 | 4    | 54               | 2.55%         | 2.33%        |
|                      | [Google Drive](https://drive.google.com/file/d/1pSWexWq21iglI1P-qmISZKGb9Qsx0uzG/view?usp=sharing)       | [Google Drive](https://drive.google.com/file/d/1seEX9D_K09byNXDNs5BO6eSY4jXpOSm6/view?usp=sharing) | 42   | 54               | **2.53%**     | **2.22%**    |
|                      | -                       | -                 | 420  | 75               | 2.57%         | 2.27%        |
|                      | -                       | -                 |      | **Best (Mean):** | 2.53% (2.55%) | 2.22% (2.27%) |
| WavLM_Nes2Net_X      | -                       | -                 | 4    | 75               | 2.53%         | 2.29%        |
|                      | [Google Drive](https://drive.google.com/file/d/1EynkhacBVdUvami7pWX8fyUQnyZaRnvV/view?usp=sharing)       | [Google Drive](https://drive.google.com/file/d/1l86auTYnhETlri9u_lhbXpUQ6IWIcHar/view?usp=sharing) | 42   | 54               | 2.53%         | **2.20%**    |
|                      | [Google Drive](https://drive.google.com/file/d/1PrzfUyXQxytWlEyTTOvKLfXYzHjz3gbX/view?usp=sharing)       | [Google Drive](https://drive.google.com/file/d/1ye_UlNQigZBQm48pLzZiMOi8MTxITvpW/view?usp=sharing) | 420  | 75               | **2.48%**     | 2.22%        |
|                      | -                       | -                 |      | **Best (Mean):** | 2.48% (2.51%) | 2.20% (2.24%) |
| WavLM_Nes2Net_X_SeLU | -                       | -                 | 4    | 75               | 2.72%         | 2.40%        |
|                      | -                       | -                 | 42   | 54               | 3.07%         | 2.69%        |
|                      | [Google Drive](https://drive.google.com/file/d/1FBuzA-UBYDkjei_ByP8u7ZAm4Dn_LLSI/view?usp=drive_link)      | [Google Drive](https://drive.google.com/file/d/12Y4KRaKOLz5oDGRaHc_ZYEjE0dU4tsHK/view?usp=sharing) | 420  | 74               | **2.28%**     | **2.02%**    |
|                      | -                       | -                 |      | **Best (Mean):** | 2.28% (2.69%) | 2.02% (2.37%) |

* Only best model checkpoints are provided.



# Usage:

  1. Git clone this repo.
  2. Build environment:
     ```
     conda env create -f SVDD.yml
     ```
     or
     ```
     pip install -r requirements.txt
     ```
     * You may need to adjust the library versions according to your CUDA version and GPU spec.
     
  4. Setup S3PRL for WavLM front-end: https://github.com/s3prl/s3prl

  5. If you want to easy inference with pre-ptrained model:
     1. Download the pretrained checkpoints from above table Google Drive links. For example, WavLM_Nes2Net_X_SeLU.
     2. Run 
     ```
     CUDA_VISIBLE_DEVICES=0 python easy_inference_demo.py \
     --model_path [pretrained_model_path] \
     --file_to_test [the file to test] \
     --model_name xxxx
     ```
     Following is an example:
     ```
     CUDA_VISIBLE_DEVICES=0 python easy_inference_demo.py \
     --model_path "/data/tianchi/Nes2Net_SVDD_ckpts/WavLM_Nes2Net_X_SeLU_e74_seed420_valid0.04245662278274772.pt" \
     --file_to_test "/home/tianchi/data/SVDD2024/test_set/CtrSVDD_0115_E_0092590.flac" \
     --model_name WavLM_Nes2Net_X_SeLU
     ```
     
  6. If you want to train the model by yourself:
     check the command template in: 
          ```
          train.sh
          ```

     Following is an example:
     ```
     python train.py --base_dir /home/tianchi/data/SVDD2024/ --algo 8 --gpu_id 2 --T_max 5 --epochs 75 --lr 0.000001 --batch_size 34 \
     --agg SEA --pool_func 'mean' --dilation 1 --Nes_ratio 8 8 --SE_ratio 1 --model_name WavLM_Nes2Net_X --seed 420 \
     --foldername WavLM_SEA_Nes2Net_X_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed420
     ```
     * Change the ```--base_dir``` to your SVDD2024 dataset path. 
     * The ```--foldername``` can be set according to your preference. 
  7. If you want to test on the CtrSVDD dataset using the released pre-trained models or your own trained model:
     check the command template in: 
          ```
          eval.sh
          ```

     Following is an example:
     ```
     CUDA_VISIBLE_DEVICES=6 python eval.py --base_dir /home/tianchi/data/SVDD2024/test_set \
     --model_path "/data/tianchi/Nes2Net_SVDD_ckpts/WavLM_Nes2Net_X_e75_seed420_valid0.03192785031473534.pt" \
     --agg SEA --pool_func 'mean' --dilation 1 --Nes_ratio 8 8 --SE_ratio 1 --model_name WavLM_Nes2Net_X \
     --outputname E75_WavLM_SEA_Nes2Net_X_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed420
     ```
     * Change the ```--base_dir``` to your SVDD2024 dataset **testset** path. 
     * Change the ```--model_path``` to your path of the checkpoint to test. 
     The default path of the model trained by our script is ```logs/[outputname]/[YYYYMMDD]-[6digits]/checkpoints/model_[epoch]_EER_[valid EER].pt```. 
     You may use the checkpoint with the smallest validation EER for testing.
     * Change the ```--agg --pool_func --dilation --Nes_ratio --SE_ratio --model_name``` to match your training setting. 
     If you are using the pretrained model, these configs are available in  ```eval.sh```. 
     To get the final result of EER and minDCF, as well as detailed results of each sub-trials:
     ```
     python EER_minDCF.py --labels_file [path to the CtrSVDD test set label txt] \
     --path [path to the score file generated by above command]
     ``` 
     For example:
     ```
     python EER_minDCF.py --labels_file '/home/tianchi/data/SVDD2024/test.txt' \
     --path scores/E75_WavLM_SEA_Nes2Net_X_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed420.txt
     ```
     Example output:
     ```
     ---------------------------------------------------------
     dataset m4singer - EER: 2.4536%  minDCF: 0.024288
     dataset kising - EER: 8.6851%  minDCF: 0.085662
     ---------------------------------------------------------
     excluding A14 only, #: 67579
     - EER: 2.2230%  minDCF: 0.022174
          ---------------------------------------------------------
     excluding both acesinger and A14, #: 64734
     - EER: 2.4782%  minDCF: 0.024745
     (atkID A09) - EER: 1.2288%  minDCF: 0.011929
     (atkID A10) - EER: 0.6305%  minDCF: 0.006173
     (atkID A11) - EER: 2.0893%  minDCF: 0.018279
     (atkID A12) - EER: 5.2686%  minDCF: 0.051162
     (atkID A13) - EER: 0.8284%  minDCF: 0.008284
     ---------------------------------------------------------
     ```
       Also note that above hyperparameters for Nes2Net series models and AASIST series models are different. 
          Please check the examples in ```train.sh``` and ```eval.sh```. 


# Reference Repo
Thanks for following open-source projects:
1. wav2vec2 + AASIST & Rawboost: https://github.com/TakHemlata/SSL_Anti-spoofing Paper: [[model]](https://arxiv.org/abs/2202.12233), [[Rawboost]](https://arxiv.org/abs/2202.12233)
2. SEA aggregation: https://github.com/Anmol2059/SVDD2024 Paper: [[SEA]](https://arxiv.org/abs/2409.02302)
3. AttM aggregation: https://github.com/pandarialTJU/AttM_INTERSPEECH24 Paper: [[AttM]](https://arxiv.org/abs/2406.10283v1)
4. WavLM pretrained model is from S3PRL: https://github.com/s3prl/s3prl

# Cite
```  
To add
```
