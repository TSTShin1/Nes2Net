import argparse
import os
import torch
from tqdm import tqdm
import librosa
import numpy as np

def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def main(args):
    path = args.base_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create the model

    if args.model_name == 'WavLM_Nes2Net':  # the proposed Nes2Net-X with SeLU activation
        from models.WavLM_Nes2Net import WavLM_Nes2Net_noRes as Model
    elif args.model_name == 'WavLM_Nes2Net_X':  # the proposed Nes2Net-X with SeLU activation
        from models.WavLM_Nes2Net_X import WavLM_Nes2Net_noRes_w_allT as Model
    elif args.model_name == 'WavLM_Nes2Net_X_SeLU':  # the proposed Nes2Net-X with SeLU activation
        from models.WavLM_Nes2Net_X_SeLU import WavLM_Nes2Net_SE_cat_SeLU as Model
    else:
        raise ValueError
    model = Model(args, device).to(device)

    # if torch.cuda.device_count() > 1: # suggest to use 1 GPU for inference only.
    #     print(f"Using DataParallel on {args.gpu_list} GPUs.")
    #     model = nn.DataParallel(model, device_ids=args.gpu_list)

    # Load the state dict of the saved model
    print(f"Loading model from: {args.model_path}")
    saved_state_dict = torch.load(args.model_path, map_location=device)

    # Get the state dict of the current model
    model_state_dict = model.state_dict()

    # Filter the saved state dict to only include keys that exist in the current model's state dict
    filtered_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_state_dict}

    # Update the current model's state dict with the filtered state dict
    model_state_dict.update(filtered_state_dict)

    # Load the updated state dict into the model
    model.load_state_dict(model_state_dict)
    print("Model loaded successfully.")

    model.eval()
    scores_out = args.output_path
    os.makedirs(scores_out, exist_ok=True)

    with torch.no_grad():
        for filename in tqdm(sorted(os.listdir(path)), desc=f"Testing"):
            if filename.lower().endswith(".wav") or filename.lower().endswith(
                    ".flac"):  # in-the-wild is wav, SVDD uses flac
                audio_path = os.path.join(path, filename)
                audio, _ = librosa.load(audio_path, sr=16000, mono=True)
                if args.test_mode == '4s':
                    audio = pad(audio, 64000)
                x = torch.tensor(audio).unsqueeze(0).to(device)
                pred = model(x)
                file_basename = os.path.splitext(filename)[0]
                with open(os.path.join(scores_out, f'{args.outputname}.txt'), "a") as f:
                    f.write(f"{file_basename} {pred.item()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--model_name", type=str, required=True,
                        choices=['WavLM_Nes2Net', 'WavLM_Nes2Net_X', 'WavLM_Nes2Net_X_SeLU'],
                        help="the type of the model, see choices")
    parser.add_argument("--agg", type=str, default='SEA', choices=['SEA', 'AttM', 'WeightedSum'],
                        help="the aggregation method for SSL")
    parser.add_argument("--dilation", type=int, default=1, help="dilation")
    parser.add_argument("--pool_func", type=str, default='mean', choices=['mean', 'ASTP'],
                        help="pooling function, choose from mean and ASTP")
    parser.add_argument("--SE_ratio", type=int, nargs='+', default=[1], help="SE downsampling ratio in the bottleneck")
    parser.add_argument("--Nes_ratio", type=int, nargs='+', default=[8, 8], help="Nes_ratio, from outer to inner")
    parser.add_argument("--AASIST_scale", type=int, default=32, choices=[24, 32, 40, 48, 56, 64, 96],
                        help="the sacle of AASIST")
    # test config
    parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the test dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="The path of the model checkpoint to test.")
    parser.add_argument("--outputname", type=str, required=True, help="The outputname to use.")
    parser.add_argument("--test_mode", type=str, default='full', choices=['4s', 'full'],
                        help="Test using either the first 4 seconds or the full length. If the file is shorter than 4s, padding will be applied.")
    parser.add_argument("--output_path", type=str, default="scores",
                        help="The path of the output folder for the scoring file.")
    # deprecated config
    # parser.add_argument("--Nes_N", type=int, default=1, help="Number of nes2net blocks")
    # parser.add_argument("--dim_reduc", type=int, default=128, help="the dimension reduce by FFN.")
    # parser.add_argument("--BTN_ratio", type=float, default=0.6, help="bottlenect ratio")
    # parser.add_argument("--num_stack", type=int, default=3, help="Number of Nes2Net blocks to stacked")
    args = parser.parse_args()
    main(args)
