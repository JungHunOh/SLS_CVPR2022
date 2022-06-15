import torch
import torch.nn as nn
import model.ops as ops
from model import common

def make_model(args, parent=False):
    return Net(args)

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1, conv=None, args=None):
        super(Block, self).__init__()
        

        self.b1 = ops.ResidualBlock(64, 64, args)
        self.b2 = ops.ResidualBlock(64, 64, args)
        self.b3 = ops.ResidualBlock(64, 64, args)

        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0, conv=conv, args=args)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0, conv=conv, args=args)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0, conv=conv, args=args)


    def forward(self, x):
        c0 = o0 = x
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        scale = args.scale[0]

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        if args.sls: self.conv = common.SparseConv
        else: self.conv = 'default'

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = Block(64, 64, conv=self.conv, args=args)
        self.b2 = Block(64, 64, conv=self.conv, args=args)
        self.b3 = Block(64, 64, conv=self.conv, args=args)
         
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0, conv=self.conv, args=args)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0, conv=self.conv, args=args)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0, conv=self.conv, args=args)

        self.upsample = common.Upsampler(self.conv, scale, 64, act='relu', args=args)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
                
    def forward(self, x):
        x = self.sub_mean(x)
        
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3)

        out = self.exit(out)
        out = self.add_mean(out)

        return out

    def compute_costs(self):
        pruned_costs = 0
        original_costs = 0
        for m in self.modules():
            if isinstance(m, self.conv):
                r = m.compute_costs()
                pruned_costs += r[0]
                original_costs += r[1]

        return pruned_costs, original_costs

    def fix_scores(self):
        for m in self.modules():
            if isinstance(m, self.conv):
                m.fix_scores()
