"""Configurable compact variants of the WiFlow-STD pose model (ADR-152 efficiency sweep).

This is a parameterized copy of upstream models/{pose_model,tcn,convnet,attention}.py
(DY2434/WiFlow @ 06899d29, Apache-2.0). upstream/ is NOT modified. Deviations from
upstream, all forced by shrinking channels and documented per variant in run_sweep.py:

1. TCN grouped-conv groups: upstream hardcodes groups=20, which does not divide
   the compact channel counts (e.g. 270, 135, 85). Rule here:
   - groups_mode='gcd20': per-conv groups = gcd(channels, 20)  (== 20 wherever
     upstream's choice is valid, incl. the 540-ch input conv; falls back to the
     largest common divisor with 20 otherwise).
   - groups_mode='depthwise': groups = channels (tiny variant only).
2. Conv2d downsampling strides: upstream uses 4 stride-(1,2) blocks because
   240/2^4 = 15 == n_keypoints. With smaller TCN output widths that would leave
   <15 rows and AdaptiveAvgPool2d((15,1)) would duplicate rows across keypoints.
   Rule: halve the width only while the result stays >= 15 (stride-2 blocks
   first, stride-1 after). Full model: 240 -> 4 halvings = upstream exactly.
3. input_pw_groups (tiny only): the dense 540->c pointwise + residual downsample
   in TCN block 1 cost 2*540*c params (a ~117k floor that alone exceeds the
   tiny <100k budget). tiny groups these two convs (groups=4; 4 | gcd(540, 68)).
4. Decoder mid-channels: upstream 64->32; here c_last -> max(c_last // 2, 4).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def tcn_groups(channels: int, mode: str) -> int:
    if mode == 'depthwise':
        return channels
    if mode == 'gcd20':
        return math.gcd(channels, 20)
    raise ValueError(mode)


# ---------------------------------------------------------------- TCN (copy of tcn.py)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class CompactGroupedTemporalBlock(nn.Module):
    """Upstream InnerGroupedTemporalBlock with parameterized groups."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2, groups_mode='gcd20', pw_groups=1):
        super().__init__()
        g_in = tcn_groups(n_inputs, groups_mode)
        g_out = tcn_groups(n_outputs, groups_mode)
        self.groups = (g_in, g_out)
        self.pw_groups = pw_groups

        self.conv1_group = nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=stride,
                                     padding=padding, dilation=dilation,
                                     groups=g_in, bias=False)
        self.chomp1 = Chomp1d(padding) if padding > 0 else nn.Identity()
        self.bn1_group = nn.BatchNorm1d(n_inputs)
        self.relu1_group = nn.SiLU(inplace=True)

        self.conv1_pw = nn.Conv1d(n_inputs, n_outputs, 1, groups=pw_groups, bias=False)
        self.bn1_pw = nn.BatchNorm1d(n_outputs)
        self.relu1_pw = nn.SiLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2_group = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=1,
                                     padding=padding, dilation=dilation,
                                     groups=g_out, bias=False)
        self.chomp2 = Chomp1d(padding) if padding > 0 else nn.Identity()
        self.bn2_group = nn.BatchNorm1d(n_outputs)
        self.relu2_group = nn.SiLU(inplace=True)

        self.conv2_pw = nn.Conv1d(n_outputs, n_outputs, 1, bias=False)
        self.bn2_pw = nn.BatchNorm1d(n_outputs)
        self.relu2_pw = nn.SiLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, 1, groups=pw_groups, bias=False),
            nn.BatchNorm1d(n_outputs)
        ) if n_inputs != n_outputs else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)
        out = self.conv1_group(x)
        out = self.chomp1(out)
        out = self.bn1_group(out)
        out = self.relu1_group(out)
        out = self.conv1_pw(out)
        out = self.bn1_pw(out)
        out = self.relu1_pw(out)
        out = self.dropout1(out)
        out = self.conv2_group(out)
        out = self.chomp2(out)
        out = self.bn2_group(out)
        out = self.relu2_group(out)
        out = self.conv2_pw(out)
        out = self.bn2_pw(out)
        out = self.relu2_pw(out)
        out = self.dropout2(out)
        return F.silu(out + res)


class CompactTemporalBlock(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2,
                 groups_mode='gcd20', input_pw_groups=1):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(CompactGroupedTemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                dropout=dropout, groups_mode=groups_mode,
                pw_groups=input_pw_groups if i == 0 else 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ------------------------------------------------------- Conv2d path (copy of convnet.py)
class AsymmetricConvBlock(nn.Module):
    """Upstream block with parameterized width stride (upstream: always (1,2))."""

    def __init__(self, in_channels, out_channels, dropout=0.3, stride_w=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3),
                      stride=(1, stride_w), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=(1, stride_w), bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.block(x) + self.downsample(x))


class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.block(x) + self.downsample(x))


# ----------------------------------------------------- attention (verbatim attention.py)
class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.stride = stride
        self.bias = bias
        self.width = width
        self.qkv_transform = nn.Conv1d(in_planes, out_planes * 3, kernel_size=1,
                                       stride=1, padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 3)
        self.bn_similarity = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes)
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)
        nn.init.normal_(self.qkv_transform.weight.data, 0, math.sqrt(1. / self.in_planes))

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)
        qkv = self.bn_qkv(self.qkv_transform(x))
        qkv = qkv.reshape(N * W, 3, self.out_planes, H).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.reshape(N * W, self.groups, self.group_planes, H)
        k = k.reshape(N * W, self.groups, self.group_planes, H)
        v = v.reshape(N * W, self.groups, self.group_planes, H)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        qk = self.bn_similarity(qk)
        similarity = F.softmax(qk, dim=-1)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sv = sv.reshape(N * W, self.out_planes, H)
        out = self.bn_output(sv)
        out = out.view(N, W, self.out_planes, H)
        if self.width:
            out = out.permute(0, 2, 1, 3)
        else:
            out = out.permute(0, 2, 3, 1)
        if self.stride > 1:
            out = self.pooling(out)
        return out


class DualAxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, stride=1, bias=False):
        super().__init__()
        self.width_axis = AxialAttention(in_planes, out_planes, groups, stride, bias, width=True)
        self.height_axis = AxialAttention(out_planes, out_planes, groups, stride, bias, width=False)

    def forward(self, x):
        return self.height_axis(self.width_axis(x))


# --------------------------------------------------------------- full model
def compute_strides(width: int, n_blocks: int, target: int = 15):
    """Halve width while result stays >= target (upstream: 240 -> 4 halvings -> 15)."""
    strides = []
    for _ in range(n_blocks):
        nxt = (width + 1) // 2  # conv k=3 s=2 p=1: out = ceil(in/2)
        if nxt >= target:
            strides.append(2)
            width = nxt
        else:
            strides.append(1)
    return strides, width


class CompactWiFlowPoseModel(nn.Module):
    """Parameterized upstream WiFlowPoseModel.

    Upstream config == tcn_channels=[540,440,340,240], conv_channels=[8,16,32,64],
    attn_groups=8, groups_mode='gcd20' (gcd(c,20)==20 for all upstream channels),
    input_pw_groups=1 -> identical architecture, 2,225,042 params.
    """

    def __init__(self, tcn_channels, conv_channels, attn_groups,
                 groups_mode='gcd20', input_pw_groups=1, dropout=0.3,
                 num_subcarriers=540, num_keypoints=15):
        super().__init__()
        self.tcn = CompactTemporalBlock(
            num_inputs=num_subcarriers, num_channels=tcn_channels, kernel_size=3,
            dropout=dropout, groups_mode=groups_mode, input_pw_groups=input_pw_groups)

        self.up = ConvBlock1(1, conv_channels[0])

        strides, self.final_width = compute_strides(
            tcn_channels[-1], len(conv_channels), target=num_keypoints)
        self.conv_strides = strides
        self.residual_blocks = nn.ModuleList()
        in_channels = conv_channels[0]
        for out_channels, s in zip(conv_channels, strides):
            self.residual_blocks.append(
                AsymmetricConvBlock(in_channels, out_channels, stride_w=s))
            in_channels = out_channels

        c_last = conv_channels[-1]
        self.attention = DualAxialAttention(c_last, c_last, groups=attn_groups)

        c_mid = max(c_last // 2, 4)
        self.decoder = nn.Sequential(
            nn.Conv2d(c_last, c_mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.SiLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((num_keypoints, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # [B, 540, 20]
        x = self.tcn(x)                          # [B, C_tcn, 20]
        x = x.transpose(1, 2).unsqueeze(1)       # [B, 1, 20, C_tcn]
        x = self.up(x)
        for block in self.residual_blocks:
            x = block(x)                         # [B, C_conv, 20, W']
        x = x.permute(0, 1, 3, 2)                # [B, C_conv, W', 20]
        x = self.attention(x)
        x = self.decoder(x)                      # [B, 2, W', 20]
        x = self.avg_pool(x).squeeze(-1)         # [B, 2, 15]
        return x.transpose(1, 2)                 # [B, 15, 2]


def describe(model: 'CompactWiFlowPoseModel'):
    params = sum(p.numel() for p in model.parameters())
    tcn_g = [blk.groups for blk in model.tcn.network]
    return {'params': params, 'tcn_groups_per_block': tcn_g,
            'conv_strides': model.conv_strides, 'final_width': model.final_width}
