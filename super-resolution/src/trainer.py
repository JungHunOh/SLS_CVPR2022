import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

from model import common

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

        self.loss_annealing_coeff = 1
        self.last2_pruned_ratios = []
        self.reach_budget = False

        self.best_psnr = 0
        self.best_epoch = 0

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        if self.args.sls:
            for m in self.model.modules():
                if isinstance(m, common.SparseConv):
                    m.update_pruning_units()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)

            if self.args.sls and not self.reach_budget:
                pruned_costs, original_costs = self.model.model.compute_costs()

                self.reach_budget = (pruned_costs - original_costs * self.args.target_budget) < 0

                if self.reach_budget:
                    self.model.model.fix_scores() 

                loss += self.args.lamb_reg * self.loss_annealing_coeff * pruned_costs

            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

        if self.args.sls:
            if self.reach_budget:
                print('Target budget is reached')
                pruned_costs, original_costs = self.model.model.compute_costs()
            else: # reg loss annealing
                current_pruned_ratio = original_costs / pruned_costs.item()
                self.current_pruned_comp = pruned_costs.item()

                self.last2_pruned_ratios.append(current_pruned_ratio)

                if epoch > 1 and epoch % 2 == 0:
                    pruned_rate_change = self.last2_pruned_ratios[-1] - self.last2_pruned_ratios[0]
                    if pruned_rate_change < self.args.annealing_thr:
                        self.loss_annealing_coeff *= self.args.alpha
                    self.last2_pruned_ratios = []
                    
            print('--Computational Costs--')
            print(f'Before pruning: {round(original_costs/1e9,2)} GMACs, After pruning: {round(pruned_costs.item()/1e9,2)} GMACs (w.r.t {self.args.patch_size}x{self.args.patch_size} image)')

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        if self.args.sls:
            for m in self.model.modules():
                if isinstance(m, common.SparseConv):
                    m.get_fixed_mask()
                    m.reassign_scores()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    
                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)

                if self.args.sls:
                    if epoch > 200:
                        psnr = self.ckp.log[-1, idx_data, idx_scale].clone()
                        if self.best_psnr < psnr:
                            self.best_psnr = psnr
                            self.best_epoch = epoch
                    else:
                        self.best_psnr = 0
                        self.best_epoch = 0
                else:
                    psnr = self.ckp.log[-1, idx_data, idx_scale].clone()
                    if self.best_psnr < psnr:
                        self.best_psnr = psnr
                        self.best_epoch = epoch

                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        self.best_psnr,
                        self.best_epoch
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(self.best_epoch == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

