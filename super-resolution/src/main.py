import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

import os
from model import common

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def main():
    global model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            
            if args.sls:
                _model.model.load_state_dict(torch.load(f'./pretrained/carn_x4_pretrained.pt'), strict=False)

            if args.pretrained_dir is not None:
                sd = torch.load(args.pretrained_dir)
                _model.model.load_state_dict(sd, strict=False)
            
            if args.compute_costs:
                utility.compute_costs(_model, args)
                return
            
            while not t.terminate():
                t.train()
                t.test()
            
            checkpoint.done()

if __name__ == '__main__':
    main()