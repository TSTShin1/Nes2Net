import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import datetime
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasetsrawboost_TC import SVDD2024
from utils import seed_worker, set_seed, compute_eer

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, use_logits=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_logits = use_logits
        
    def forward(self, logits, targets):
        if self.use_logits:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def check_condition(i, m):
    # validate the checkpoint of the epoch with the minimum learning rate,
    # as well as the checkpoints of the preceding and following epochs.
    return i == 1 or (i > 2 * m + 1 and (i - m) % (2 * m) in (2 * m - 1, 0, 1))

def main(args):
    # Set the seed for reproducibility
    set_seed(args.seed)
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Create the dataset
    path = args.base_dir
    train_dataset = SVDD2024(path, partition="train",args=args, algo=args.algo)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker)

    print('loading dataset using datasetsrawboost_TC, where algo is not used, file is not cut or pad and batch size is 1 for validation')
    dev_dataset = SVDD2024(path, partition="dev",args=args, algo=0) # not using
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    
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

    Total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('---Total params:', Total_params)
    SSL_params = sum(p.numel() for p in model.ssl_model.parameters() if p.requires_grad)
    print('---SSL params (w/ aggregation):', SSL_params)
    print('---Backend params:', Total_params - SSL_params)


    # Create the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr)

    # Create the directory for the logs
    log_dir = os.path.join(args.log_dir, args.foldername)
    os.makedirs(log_dir, exist_ok=True)
    
    # get current time
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, current_time)
    os.makedirs(log_dir, exist_ok=True)

    # Create the summary writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create the directory for the checkpoints
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config for reproducibility
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(str(vars(args)))
        
    criterion = BinaryFocalLoss()
    
    best_val_eer = 1.0
    start_epoch = 1
    # try to resume training:
    if args.resume_checkpoint_path != 'unavailable':
        try:
            checkpoint = torch.load(args.resume_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print('Scuessfully resume training from saved checkpoint. Start from epoch', start_epoch)
        except FileNotFoundError:
            print('Cant resume training from provided checkpoint path:', args.resume_checkpoint_path)
            print('Train from scratch. Epoch:', start_epoch)

    # Train the model
    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        pos_samples, neg_samples = [], []
        
        SSL_freeze = False if epoch > args.SSL_freeze_epoch else True
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)):
            if args.debug and i > 20:
                break
            x, label, _ = batch
            x = x.to(device)
            label = label.to(device)
            soft_label = label.float() * 0.9 + 0.05
            pred = model(x, SSL_freeze=SSL_freeze)
            loss = criterion(pred, soft_label.unsqueeze(1))
            pos_samples.append(pred[label == 1].detach().cpu().numpy())
            neg_samples.append(pred[label == 0].detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
        scheduler.step()
        writer.add_scalar("LR/train", scheduler.get_last_lr()[0], epoch * len(train_loader) + i)
        writer.add_scalar("EER/train", compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0], epoch)
        print('LR:', scheduler.get_last_lr()[0], 'last iter loss:', loss.item())

        checkpoint_for_resume = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint_for_resume, os.path.join(checkpoint_dir, f"newest_checkpoint_for_resume.pth"))
        print('Epoch {}: Saved newest end of epoch checkpoint for resuming training purpose as newest_checkpoint_for_resume.pth,'.format(epoch))

        if check_condition(epoch, args.T_max):
            model.eval()
            val_loss = 0
            pos_samples, neg_samples = [], []
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dev_loader, desc=f"Validation", dynamic_ncols=True)):
                    if args.debug and i > 20:
                        break
                    x, label, _ = batch
                    x = x.to(device)
                    label = label.to(device)
                    pred = model(x)
                    soft_label = label.float() * 0.9 + 0.05
                    loss = criterion(pred, soft_label.unsqueeze(1))
                    pos_samples.append(pred[label == 1].detach().cpu().numpy())
                    neg_samples.append(pred[label == 0].detach().cpu().numpy())
                    val_loss += loss.item()
                val_loss /= len(dev_loader)
                val_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("EER/val", val_eer, epoch)
                print('epoch:', epoch, 'EER:', val_eer)
                if val_eer < best_val_eer:
                    best_val_eer = val_eer
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model.pt"))
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{epoch}_EER_{val_eer}.pt"))

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
    # train config
    parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset.")
    parser.add_argument("--epochs", type=int, default=75, help="The number of epochs to train.")
    parser.add_argument("--resume_checkpoint_path", type=str, default='unavailable', help="resume_checkpoint_path of newest_checkpoint_for_resume.pth")
    parser.add_argument("--gpu_id", type=int, default=0, help="The GPU to use.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--foldername", type=str, default="0629_SE_algo8_Tmax4", help="The foldername to use.")
    parser.add_argument("--batch_size", type=int, default=34, help="The batch size for training.")
    parser.add_argument("--num_workers", type=int, default=12, help="The number of workers for the data loader.")
    parser.add_argument("--log_dir", type=str, default="logs", help="The directory for the logs.")
    parser.add_argument("--T_max", type=int, default=5, help="T_max of cosine.")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--SSL_freeze_epoch", type=int, default=-1,
                        help="freeze the SSL for [SSL_freeze_epoch] epochs. deafult = -1, fine-tune since the beginning.")
    parser.add_argument("--lr", type=float, default=0.000001, help="learning rate max")
    parser.add_argument("--min_lr", type=float, default=0.000000001, help="learning rate min")

    # deprecated config
    # parser.add_argument("--BTN_ratio", type=float, default=0.6, help="bottlenect ratio")
    # parser.add_argument("--Nes_N", type=int, default=1, help="Number of nes2net blocks")
    # parser.add_argument("--dim_reduc", type=int, default=128, help="the dimension reduce by FFN.")
    # parser.add_argument("--num_stack", type=int, default=3, help="Number of Nes2Net blocks to stacked")

    # rawboost config
    parser.add_argument('--algo', type=int, default=8,
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, \
                        3: SSI_additive_noise, 4: series algo (1+2+3), 5: series algo (1+2), 6: series algo (1+3), \
                        7: series algo(2+3), 8: parallel algo(1,2), 9: parallel algo(1,2,3).10: parallel algo(1,2) with possibility [default=0]')
    parser.add_argument('--LnL_ratio', type=float, default=1.0, 
                    help='This is the possibility to activate LnL, which will only be used when algo>=10.')
    parser.add_argument('--ISD_ratio', type=float, default=1.0, 
                    help='This is the possibility to activate ISD, which will only be used when algo>=10.')
    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')

    args = parser.parse_args()
    main(args)