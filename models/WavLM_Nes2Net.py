import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils
import math

___author__ = "Tianchi Liu"
__email__ = "tianchi_liu@u.nus.edu"


class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        cp_path = 'xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()      

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
                
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

class SEModule(nn.Module):
    def __init__(self, channels, SE_ratio=8):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // SE_ratio, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels // SE_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8, SE_ratio=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes,SE_ratio)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out


class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super(ASTP, self).__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat([mean, std], dim=1)


class Nested_Res2Net_TDNN(nn.Module):

    def __init__(self, Nes_ratio=[8, 8], input_channel=1024, dilation=2, pool_func='mean', SE_ratio=[8]):

        super(Nested_Res2Net_TDNN, self).__init__()
        self.Nes_ratio = Nes_ratio[0]
        assert input_channel % Nes_ratio[0] == 0
        C = input_channel // Nes_ratio[0]
        self.C = C
        Build_in_Res2Nets = []
        bns = []
        for i in range(Nes_ratio[0] - 1):
            Build_in_Res2Nets.append(Bottle2neck(C, C, kernel_size=3, dilation=dilation, scale=Nes_ratio[1], SE_ratio=SE_ratio[0]))
            bns.append(nn.BatchNorm1d(C))
        self.Build_in_Res2Nets = nn.ModuleList(Build_in_Res2Nets)
        self.bns = nn.ModuleList(bns)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.pool_func = pool_func
        if pool_func == 'mean':
            self.fc = nn.Linear(1024, 1)
        elif pool_func == 'ASTP':
            self.pooling = ASTP(in_dim=input_channel, bottleneck_dim=128, global_context_att=False)
            self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        spx = torch.split(x, self.C, 1)
        for i in range(self.Nes_ratio - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.Build_in_Res2Nets[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[-1]), 1)
        out = self.bn(out)
        out = self.relu(out)
        if self.pool_func == 'mean':
            out = torch.mean(out, dim=-1)
        elif self.pool_func == 'ASTP':
            out = self.pooling(out)
        out = self.fc(out)
        return out


class WavLM_Nes2Net_noRes(nn.Module):
    def __init__(self, args, device):
        super(WavLM_Nes2Net_noRes, self).__init__()
        self.device = device

        # create network WavLM
        self.ssl_model = SSLModel(self.device)
        # self.fc = nn.Linear(1024, 128)
        self.Nested_Res2Net_TDNN = Nested_Res2Net_TDNN(Nes_ratio=args.Nes_ratio, input_channel=1024,
                                                       dilation=args.dilation, pool_func=args.pool_func, SE_ratio=args.SE_ratio)

    def forward(self, x, SSL_freeze=False):
        x = x.to(self.device)

        # Pre-trained WavLM model fine-tuning
        if SSL_freeze:
            with torch.no_grad():
                x_ssl_feat = self.ssl_model.extract_feat(x)
        else:
            x_ssl_feat = self.ssl_model.extract_feat(x)
        x_ssl_feat = x_ssl_feat.permute(0, 2, 1)  # bz 1024 T
        output = self.Nested_Res2Net_TDNN(x_ssl_feat)

        return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--agg", type=str, default='SEA', choices=['SEA', 'AttM', 'WeightedSum'],
                        help="the aggregation method for SSL")
    parser.add_argument("--dilation", type=int, default=1, help="dilation")
    parser.add_argument("--pool_func", type=str, default='mean', choices=['mean', 'ASTP'],
                        help="pooling function, choose from mean and ASTP")
    parser.add_argument("--SE_ratio", type=int, nargs='+', default=[1], help="SE downsampling ratio in the bottleneck")
    parser.add_argument("--Nes_ratio", type=int, nargs='+', default=[8, 8], help="Nes_ratio, from outer to inner")

    args = parser.parse_args()
    model = WavLM_Nes2Net_noRes(args=args, device='cpu')
    x = torch.rand((4, 32000)).to('cpu')
    model = model.to('cpu')
    y = model(x)
    print(y)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)
    trainable_params = sum(p.numel() for p in model.ssl_model.parameters() if p.requires_grad)
    print(trainable_params)
    trainable_params = sum(p.numel() for p in model.Nested_Res2Net_TDNN.parameters() if p.requires_grad)
    print(trainable_params)
    # trainable_params = sum(p.numel() for p in model.fc.parameters() if p.requires_grad)
    # print(trainable_params)

    from fvcore.nn import FlopCountAnalysis

    # Profiling the model
    x = torch.rand((1, 1024, 200)).to('cpu')
    flops = FlopCountAnalysis(model.Nested_Res2Net_TDNN, x)
    print(f"FLOPs: {flops.total()} ({flops.total() / 1e9} GFLOPs)")

    from ptflops import get_model_complexity_info

    input_shape = (1024, 200)
    macs, params = get_model_complexity_info(model.Nested_Res2Net_TDNN, input_shape, as_strings=True,
                                             print_per_layer_stat=True)
    print(f"MACs: {macs}")
    print(f"Parameters: {params}")
