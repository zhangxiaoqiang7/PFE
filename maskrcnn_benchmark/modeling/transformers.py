import torch
import torch.nn.functional as F
from torch import nn

import math,copy
import pdb

def _init_weight(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            torch.nn.init.kaiming_normal_(m.weight)
            # torch.nn.init.normal_(m.weight, std=0.01)
            # torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
    / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
                
class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        with_bn=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation,)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod, 
                                          nn.BatchNorm2d(int(n_filters)), 
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
        
        _init_weight(self.modules())

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.1, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        _init_weight(self.modules())
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
        [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
        dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
        .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
        
        
class pyramidPooling(nn.Module):
    def __init__(self, cfg):
        super(pyramidPooling, self).__init__()
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        pool_sizes=cfg.MODEL.TRANSFORMER.POOL_SIZES
        out_channels = int(in_channels / len(pool_sizes))
        self.paths = []
        for pool_size in pool_sizes:
            self.paths.append(nn.Sequential(nn.AdaptiveAvgPool2d((pool_size, pool_size)),
                              conv2DBatchNormRelu(in_channels,out_channels,1,1,0,bias=False)))

        self.pool_modules = nn.ModuleList(self.paths)
        self.conv1x1 = conv2DBatchNormRelu(in_channels*2,in_channels,1,1,0,bias=False)

    def forward(self, x):
        x = x[0]
        output_slices = [x]
        for module in self.pool_modules:
            out = module(x)
            out = F.upsample(out, size=x.size()[2:], mode="bilinear", align_corners=True)
            output_slices.append(out)
        x = torch.cat(output_slices, dim=1)
        x = self.conv1x1(x)
        return [x]

class contextPooling(nn.Module):
    def __init__(self, cfg):
        super(contextPooling, self).__init__()
        self.cfg = cfg.clone()
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        kernel_sizes=cfg.MODEL.TRANSFORMER.KERNEL_SIZES
        out_channels = int(in_channels / len(kernel_sizes))
        self.paths = []
        for k_size in kernel_sizes:
            p = int((k_size-1)/2)
            self.paths.append(nn.Sequential(nn.AvgPool2d(k_size,stride=1,padding=(p,p),count_include_pad=False),
                              conv2DBatchNormRelu(in_channels,out_channels,3,1,2,bias=False,dilation=2)))

        self.pool_modules = nn.ModuleList(self.paths)
        self.conv1x1 = conv2DBatchNormRelu(out_channels*len(kernel_sizes),in_channels,1,1,0,bias=False)
        #self.conv1x1 = BasicConv(out_channels*len(kernel_sizes), in_channels, kernel_size=1, stride=1, padding=0,relu=False,bn=True)
        #self.relu = nn.ReLU()

    def forward(self, x):
        x = x[0]
        #x0 = x
        n,c,h,w = x.size()
        #2x1024x36x63
        output_slices = []
        for module in self.pool_modules:
            out = module(x)
            #out = F.upsample(out, size=x.size()[2:], mode="bilinear", align_corners=True)
            output_slices.append(out)
        x = torch.cat(output_slices, dim=1)
        x = self.conv1x1(x)
        #x = self.relu(x+x0)
        return [x]
            
class ASPP_module(nn.Module):
    def __init__(self, in_channels, out_channels, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = conv2DBatchNormRelu(in_channels,out_channels,
                                  kernel_size,1,padding,dilation=rate,bias=False)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x
    
class ASPP(nn.Module):
    def __init__(self, cfg):
        super(ASPP, self).__init__()
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        rates = cfg.MODEL.TRANSFORMER.RATES
        out_channels = int(in_channels / len(rates))
        self.aspp_modules = nn.ModuleList([ASPP_module(in_channels,out_channels,r) for r in rates])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             conv2DBatchNormRelu(in_channels,out_channels,1,1,0,bias=False))
        self.conv1x1 = conv2DBatchNormRelu(out_channels*(len(rates)+1),in_channels,1,1,0,bias=False)
        
    def forward(self, x):
        x = x[0]
        outs = [a(x) for a in self.aspp_modules]
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=outs[0].size()[2:], mode='bilinear', align_corners=True)
        outs.append(x5)
        x = torch.cat(outs, dim=1)
        x = self.conv1x1(x)
        return [x]
        

        
class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 1.0):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        #pdb.set_trace()
        x = x[0]
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return [out]
def RFB_module(cfg):
     in_planes = cfg.MODEL.BACKBONE.OUT_CHANNELS
     return BasicRFB_a(in_planes,in_planes)
        
_FEATURE_EXTRACTORS = {
    "ASPP": ASPP,
    "PSP": pyramidPooling,
    "CP": contextPooling,
    "RFB": RFB_module,
}

def build_transformers(cfg):
    func = _FEATURE_EXTRACTORS[cfg.MODEL.TRANSFORMER.TYPE]
    return func(cfg)