import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from timm.models.layers import DropPath
from functools import partial
from encoder.MidEnhance import MidEnhance, Up


def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self, channel=128, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)  # bs,k,n,c
        a = torch.sum(torch.sum(x_all, 1), 1)  # bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # bs,kc
        hat_a = hat_a.reshape(b, self.k, c)  # bs,k,c
        bar_a = self.softmax(hat_a)  # bs,k,c
        attention = bar_a.unsqueeze(-2)  # #bs,k,1,c
        out = attention * x_all  # #bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out


class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()

    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H * W, self.heads, 3 * self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (H * W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias

        return bias


class simamAttention(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simamAttention, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(self, in_channels, out_channels, n_feats=64, n_resblocks=16, res_scale=1,
                 scale=[2], no_upsampling=True, rgb_range=1, conv=default_conv):
        super(EDSR, self).__init__()

        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        kernel_size = 3
        self.scale = scale[0]
        self.no_upsampling = no_upsampling
        act = nn.ReLU(True)
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(in_channels, out_channels, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, out_channels, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(self.n_resblocks)
        ]
        m_body.append(conv(out_channels, out_channels, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if self.no_upsampling:
            self.out_dim = self.n_feats
        else:
            self.out_dim = in_channels
            # define tail module
            m_tail = [
                Upsampler(conv, self.scale, self.n_feats, act=False),
                conv(self.n_feats, in_channels, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x

        if self.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        # x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, G0=64, RDNkSize=3, RDNconfig='B', scale=2, no_upsampling=True):
        super(RDN, self).__init__()
        r = scale
        self.G0 = G0
        self.kSize = RDNkSize
        self.no_upsampling = no_upsampling

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (1, 1, 3),
            'B': (20, 6, 32),
            'C': (16, 8, 64),

        }[RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(in_channels, self.G0, self.kSize, padding=(self.kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, self.kSize, padding=(self.kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=self.G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, self.kSize, padding=(self.kSize - 1) // 2, stride=1)
        ])

        if self.no_upsampling:
            self.out_dim = self.G0
        else:
            self.out_dim = out_channels
            # Up-sampling net
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(self.G0, G * r * r, self.kSize, padding=(self.kSize - 1) // 2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(G, out_channels, self.kSize, padding=(self.kSize - 1) // 2, stride=1)
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(self.G0, G * 4, self.kSize, padding=(self.kSize - 1) // 2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, self.kSize, padding=(self.kSize - 1) // 2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, out_channels, self.kSize, padding=(self.kSize - 1) // 2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        if self.no_upsampling:
            return x
        else:
            return self.UPNet(x)


class CAB(nn.Module):
    def __init__(self, features):
        super(CAB, self).__init__()

        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen1[2].weight.data.zero_()
        self.delta_gen2[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w / s, h / s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)

        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        return low_stage, high_stage


class DSC(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1):
        super(DSC, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class AlignDecoderBlock(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels):
        super(AlignDecoderBlock, self).__init__()

        self.up = CAB(input_channels)
        self.identity_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        self.decode = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            DSC(input_channels, input_channels),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            DSC(input_channels, output_channels),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, low_feat, high_feat):
        l, h = self.up(low_feat, high_feat)
        resl = self.identity_conv(l)
        outl = self.decode(l)
        outl = outl + resl
        resh = self.identity_conv(h)
        outh = self.decode(h)
        outh = outh + resh

        return outl, outh


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """

    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        # Linear Attention
        x = self.attn(x)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"


class GatedFusion(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GatedFusion, self).__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)

        PG = x * G
        FG = y * (1 - G)

        return FG + PG


class TriConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TriConv, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels // 2, out_channels=out_channels, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.encoder1(x)

        return out


nonlinearity = partial(F.relu, inplace=True)


class MSDC2(nn.Module):
    def __init__(self, channel, features, out_features=256, sizes=(1, 2, 3, 6)):
        super(MSDC2, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self.create(features, size) for size in sizes])
        self.bottle = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=7, padding=7)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate6 = nn.Conv2d(channel, channel, kernel_size=3, dilation=7, padding=7)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def create(self, features, size):
        pool = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(pool, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        fusion = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottle(torch.cat(fusion, 1))
        x = self.relu(bottle)
        dilate1_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate4(self.dilate3(self.dilate2(self.dilate1(x))))))
        dilate4_out = nonlinearity(
            self.conv1x1(self.dilate5(self.dilate4(self.dilate3(self.dilate2(self.dilate1(x)))))))
        dilate5_out = nonlinearity(
            self.conv1x1(self.dilate6(self.dilate5(self.dilate4(self.dilate3(self.dilate2(self.dilate1(x))))))))
        dilate6_out = self.pooling(x)

        out_feature = dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out + dilate6_out

        return out_feature


class PyConv(nn.Module):
    def __init__(self, in_channels):
        super(PyConv, self).__init__()
        self.encoder1 = self._conv3x3(in_channels=in_channels, out_channels=in_channels // 4)
        self.encoder2 = self._conv5x5(in_channels=in_channels, out_channels=in_channels // 4)
        self.encoder3 = self._conv7x7(in_channels=in_channels, out_channels=in_channels // 4)
        self.encoder4 = self._conv9x9(in_channels=in_channels, out_channels=in_channels // 4)

    def _conv1x1(self, in_channels, out_channels, stride=1, bias=False):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)

    def _conv3x3(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=1)

    def _conv5x5(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False, groups=4)

    def _conv7x7(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False, groups=8)

    def _conv9x9(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4, bias=False, groups=16)

    def forward(self, x):
        d1 = self.encoder1(x)
        d2 = self.encoder2(x)
        d3 = self.encoder3(x)
        d4 = self.encoder4(x)
        out = torch.cat((d1, d2, d3, d4), dim=1)

        return out


class HDP(nn.Module):
    def __init__(self, in_channels):
        super(HDP, self).__init__()
        self.encoder1 = nn.Sequential(PyConv(in_channels=in_channels),
                                      nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False),
                                      S2Attention(channels=in_channels // 2))
        self.encoder2 = nn.Sequential(MSDC2(channel=in_channels, features=in_channels),
                                      nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False),
                                      S2Attention(channels=in_channels // 2))

    def forward(self, x):
        d1 = self.encoder1(x)
        d2 = self.encoder2(x)
        out = torch.cat((d1, d2), dim=1)
        return out


class GMSFPM(nn.Module):
    def __init__(self, in_channels):
        super(GMSFPM, self).__init__()
        channels_mid = int(in_channels // 4)

        self.channels_cond = in_channels

        self.conv_master = nn.Conv2d(self.channels_cond, in_channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(in_channels)

        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3,
                                   bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2,
                                   bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1,
                                   bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3,
                                   bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2,
                                   bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1,
                                   bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1,
                                                  bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1,
                                                  bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, in_channels, kernel_size=4, stride=2, padding=1,
                                                  bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """

        x_master1 = self.conv_master(x)
        x_master1 = self.bn_master(x_master1)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)
        x_master = x_master1 * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

        out = self.relu(x_master)

        return out, x_master1, x1_2, x2_2, x3_2


class GMSPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(GMSPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2d(inplanes, momentum=0.1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    nn.BatchNorm2d(inplanes, momentum=0.1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    nn.BatchNorm2d(inplanes, momentum=0.1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.BatchNorm2d(inplanes, momentum=0.1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            nn.BatchNorm2d(branch_planes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(branch_planes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(branch_planes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(branch_planes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear') + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out, x_list[1], x_list[2], x_list[3], x_list[4]


class SFP(nn.Module):
    def __init__(self, in_channels):
        super(SFP, self).__init__()
        self.encoder1 = GMSFPM(in_channels=in_channels)
        self.encoder2 = GMSPPM(inplanes=256, branch_planes=128, outplanes=256)
        self.encoder3 = MidEnhance(hidden_dim=256, guidance_x_dim=256, guidance_y_dim=128, nheads=4,
                                   input_resolution=(8, 8), pooling_size=(4, 4), window_size=10,
                                   attention_type='linear')
        self.up1 = Up(in_channels=64, guidance_channels=128, nheads=4, attention_type='linear', kernel_size=7)
        self.encoder4 = MidEnhance(hidden_dim=256, guidance_x_dim=64, guidance_y_dim=128, nheads=4,
                                   input_resolution=(8, 8), pooling_size=(4, 4), window_size=10,
                                   attention_type='linear')
        self.up2 = Up(in_channels=64, guidance_channels=128, nheads=4, attention_type='linear', kernel_size=7)
        self.encoder5 = MidEnhance(hidden_dim=256, guidance_x_dim=64, guidance_y_dim=128, nheads=4,
                                   input_resolution=(8, 8), pooling_size=(4, 4), window_size=10,
                                   attention_type='linear')
        self.up3 = Up(in_channels=64, guidance_channels=128, nheads=4, attention_type='linear', kernel_size=7)
        self.encoder6 = MidEnhance(hidden_dim=256, guidance_x_dim=64, guidance_y_dim=128, nheads=4,
                                   input_resolution=(8, 8), pooling_size=(4, 4), window_size=10,
                                   attention_type='linear')
        self.norm = nn.BatchNorm2d(256)
        self.att1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False),
                                  simamAttention())
        self.att2 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False),
                                  simamAttention())

    def forward(self, x):
        d11, d12, d13, d14, d15 = self.encoder1(x)
        d21, d22, d23, d24, d25 = self.encoder2(x)
        add = self.encoder3(x, d12, d22)
        d13 = self.up1(d13, d23)
        add = self.encoder4(add, d13, d23)
        d14 = self.up2(d14, d24)
        add = self.encoder5(add, d14, d24)
        d15 = self.up3(d15, d25)
        add = self.encoder6(add, d15, d25)
        add = self.norm(add)
        d11 = self.att1(d11 + add)
        d21 = self.att2(d21 + add)
        out = torch.cat((d11, d21), dim=1)
        return out


class AlignOS(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(AlignOS, self).__init__()
        self.decoder1 = AlignDecoderBlock(in_channels, in_channels)
        self.decoder2 = GatedFusion(in_channels, in_channels)
        self.decoder3 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(in_channels // 2),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels // 2, n_classes, kernel_size=1, bias=False),
                                      nn.Sigmoid())

    def forward(self, x, y):
        x1, y1 = self.decoder1(x, y)
        out = self.decoder2(x1, y1)
        out = self.decoder3(out)
        return out


class PSiam_HDSFNet(nn.Module):
    def __init__(self, ms_channels, sar_channels, n_classes):
        self.ms_channels = ms_channels
        self.sar_channels = sar_channels

        super(PSiam_HDSFNet, self).__init__()
        # Optical processing
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = EDSR(in_channels=self.ms_channels, out_channels=16, n_resblocks=4)
        self.encoder2 = EDSR(in_channels=16, out_channels=32, n_resblocks=4)
        self.encoder3 = EDSR(in_channels=32, out_channels=64, n_resblocks=4)
        self.encoder4 = EDSR(in_channels=64, out_channels=128, n_resblocks=4)
        self.encoder5 = EDSR(in_channels=128, out_channels=256, n_resblocks=4)

        self.encoder6 = HDP(in_channels=256)
        self.encoder7 = SFP(in_channels=256)

        self.decoder1 = nn.Sequential(TriConv(in_channels=512, out_channels=324),
                                      simple_attn(324, 18),
                                      nn.BatchNorm2d(324),
                                      nn.ReLU())
        self.decoder2 = nn.Sequential(TriConv(in_channels=452, out_channels=256),
                                      simple_attn(256, 16),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU())
        self.decoder3 = nn.Sequential(TriConv(in_channels=320, out_channels=196),
                                      simple_attn(196, 14),
                                      nn.BatchNorm2d(196),
                                      nn.ReLU())
        self.decoder4 = nn.Sequential(TriConv(in_channels=228, out_channels=144),
                                      simple_attn(144, 12),
                                      nn.BatchNorm2d(144),
                                      nn.ReLU())
        self.decoder5 = nn.Sequential(TriConv(in_channels=160, out_channels=16),
                                      simple_attn(16, 4),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU())
        self.skip1 = MLLABlock(dim=16, input_resolution=(256, 256), num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.skip2 = MLLABlock(dim=32, input_resolution=(128, 128), num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.skip3 = MLLABlock(dim=64, input_resolution=(64, 64), num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.skip4 = MLLABlock(dim=128, input_resolution=(32, 32), num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.skip5 = MLLABlock(dim=256, input_resolution=(16, 16), num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)

        # SAR processing
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder8 = RDN(in_channels=self.sar_channels, out_channels=16, G0=16, RDNkSize=3, RDNconfig='A', scale=2,
                            no_upsampling=True)
        self.encoder9 = RDN(in_channels=16, out_channels=32, G0=32, RDNkSize=3, RDNconfig='A', scale=2,
                            no_upsampling=True)
        self.encoder10 = RDN(in_channels=32, out_channels=64, G0=64, RDNkSize=3, RDNconfig='A', scale=2,
                             no_upsampling=True)
        self.encoder11 = RDN(in_channels=64, out_channels=128, G0=128, RDNkSize=3, RDNconfig='A', scale=2,
                             no_upsampling=True)
        self.encoder12 = RDN(in_channels=128, out_channels=256, G0=256, RDNkSize=3, RDNconfig='A', scale=2,
                             no_upsampling=True)

        self.encoder13 = HDP(in_channels=256)
        self.encoder14 = SFP(in_channels=256)

        self.decoder6 = nn.Sequential(TriConv(in_channels=512, out_channels=324),
                                      simple_attn(324, 18),
                                      nn.BatchNorm2d(324),
                                      nn.ReLU())
        self.decoder7 = nn.Sequential(TriConv(in_channels=452, out_channels=256),
                                      simple_attn(256, 16),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU())
        self.decoder8 = nn.Sequential(TriConv(in_channels=320, out_channels=196),
                                      simple_attn(196, 14),
                                      nn.BatchNorm2d(196),
                                      nn.ReLU())
        self.decoder9 = nn.Sequential(TriConv(in_channels=228, out_channels=144),
                                      simple_attn(144, 12),
                                      nn.BatchNorm2d(144),
                                      nn.ReLU())
        self.decoder10 = nn.Sequential(TriConv(in_channels=160, out_channels=16),
                                       simple_attn(16, 4),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.skip6 = MLLABlock(dim=16, input_resolution=(256, 256), num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.skip7 = MLLABlock(dim=32, input_resolution=(128, 128), num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.skip8 = MLLABlock(dim=64, input_resolution=(64, 64), num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.skip9 = MLLABlock(dim=128, input_resolution=(32, 32), num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.skip10 = MLLABlock(dim=256, input_resolution=(16, 16), num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0.,
                                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)

        self.decoder11 = AlignOS(in_channels=16, n_classes=n_classes)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)

    def forward(self, x):
        # MS processing
        x1 = x[:, 0: self.ms_channels, :, :]
        e1 = self.encoder1(x1)
        x2 = self.maxpool1(e1)
        e1 = self.skip1(e1)

        e2 = self.encoder2(x2)
        x3 = self.maxpool1(e2)
        e2 = self.skip2(e2)

        e3 = self.encoder3(x3)
        x4 = self.maxpool1(e3)
        e3 = self.skip3(e3)

        e4 = self.encoder4(x4)
        x5 = self.maxpool1(e4)
        e4 = self.skip4(e4)

        e5 = self.encoder5(x5)
        x6 = self.maxpool1(e5)
        e5 = self.skip5(e5)

        x6 = self.encoder6(x6)
        x6 = self.encoder7(x6)

        # SAR processing
        x7 = x[:, self.ms_channels: self.ms_channels + self.sar_channels, :, :]
        e6 = self.encoder8(x7)
        x8 = self.maxpool2(e6)
        e6 = self.skip6(e6)

        e7 = self.encoder9(x8)
        x9 = self.maxpool2(e7)
        e7 = self.skip7(e7)

        e8 = self.encoder10(x9)
        x10 = self.maxpool2(e8)
        e8 = self.skip8(e8)

        e9 = self.encoder11(x10)
        x11 = self.maxpool2(e9)
        e9 = self.skip9(e9)

        e10 = self.encoder12(x11)
        x12 = self.maxpool2(e10)
        e10 = self.skip10(e10)

        x12 = self.encoder13(x12)
        x12 = self.encoder14(x12)

        # MS processing2
        x6 = F.interpolate(x6, size=(e5.shape[2], e5.shape[3]), mode='bilinear', align_corners=True)
        x6 = torch.cat((x6, e5), dim=1)
        x6 = self.decoder1(x6)

        x6 = F.interpolate(x6, size=(e4.shape[2], e4.shape[3]), mode='bilinear', align_corners=True)
        x6 = torch.cat((x6, e4), dim=1)
        x6 = self.decoder2(x6)

        x6 = F.interpolate(x6, size=(e3.shape[2], e3.shape[3]), mode='bilinear', align_corners=True)
        x6 = torch.cat((x6, e3), dim=1)
        x6 = self.decoder3(x6)

        x6 = F.interpolate(x6, size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=True)
        x6 = torch.cat((x6, e2), dim=1)
        x6 = self.decoder4(x6)

        x6 = F.interpolate(x6, size=(e1.shape[2], e1.shape[3]), mode='bilinear', align_corners=True)
        x6 = torch.cat((x6, e1), dim=1)
        x6 = self.decoder5(x6)

        # SAR processing2
        x12 = F.interpolate(x12, size=(e10.shape[2], e10.shape[3]), mode='bilinear', align_corners=True)
        x12 = torch.cat((x12, e10), dim=1)
        x12 = self.decoder6(x12)

        x12 = F.interpolate(x12, size=(e9.shape[2], e9.shape[3]), mode='bilinear', align_corners=True)
        x12 = torch.cat((x12, e9), dim=1)
        x12 = self.decoder7(x12)

        x12 = F.interpolate(x12, size=(e8.shape[2], e8.shape[3]), mode='bilinear', align_corners=True)
        x12 = torch.cat((x12, e8), dim=1)
        x12 = self.decoder8(x12)

        x12 = F.interpolate(x12, size=(e7.shape[2], e7.shape[3]), mode='bilinear', align_corners=True)
        x12 = torch.cat((x12, e7), dim=1)
        x12 = self.decoder9(x12)

        x12 = F.interpolate(x12, size=(e6.shape[2], e6.shape[3]), mode='bilinear', align_corners=True)
        x12 = torch.cat((x12, e6), dim=1)
        x12 = self.decoder10(x12)

        out = self.decoder11(x6, x12)

        return out