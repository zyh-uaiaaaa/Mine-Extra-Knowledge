
from timm.utils import accuracy, AverageMeter

import logging
import time
import torch
import torch.distributed as dist
import math

from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed

class LimitedAvgMeter(object):

    def __init__(self, max_num=10, best_mode="max"):
        self.avg = 0.0
        self.num_list = []
        self.max_num = max_num
        self.best_mode = best_mode
        self.best = 0.0 if best_mode == "max" else 100.0

    def append(self, x):
        self.num_list.append(x)
        len_list = len(self.num_list)
        if len_list > 0:
            if len_list < self.max_num:
                self.avg = sum(self.num_list) / len_list
            else:
                self.avg = sum(self.num_list[(len_list - self.max_num):len_list]) / self.max_num

        if self.best_mode == "max":
            if self.avg > self.best:
                self.best = self.avg
        elif self.best_mode == "min":
            if self.avg < self.best:
                self.best = self.avg

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

class ExpressionVerification(object):

    def __init__(self, data_loader, summary_writer=None):
        self.rank: int = distributed.get_rank()
        self.highest_acc1: float = 0.0
        self.highest_acc5: float = 0.0

        self.data_loader = data_loader
        self.summary_writer = summary_writer
        self.limited_meter = LimitedAvgMeter(best_mode="max")

    def ver_test(self, model, global_step):

        logging.info("Val on RAF/AffectNet:")

        criterion = torch.nn.CrossEntropyLoss()

        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()
        for idx, (images, target, _) in enumerate(self.data_loader):
            img = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output

            expression_output, _ = model(img)

            loss = criterion(expression_output, target)
            acc1, acc5 = accuracy(expression_output, target, topk=(1, 5))

            loss = reduce_tensor(loss)
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info(
                    f'Test: [{idx}/{len(self.data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        if acc1_meter.avg > self.highest_acc1:
            self.highest_acc1 = acc1_meter.avg
        if acc5_meter.avg > self.highest_acc5:
            self.highest_acc5 = acc5_meter.avg

        self.limited_meter.append(acc1_meter.avg)

        if self.rank is 0:
            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag="expression loss", scalar_value=loss_meter.avg, global_step=global_step)
            self.summary_writer.add_scalar(tag="expression acc1", scalar_value=acc1_meter.avg, global_step=global_step)
            self.summary_writer.add_scalar(tag="expression acc5", scalar_value=acc5_meter.avg, global_step=global_step)

            logging.info('[%d]Expression Loss: %1.5f' % (global_step, loss_meter.avg))
            logging.info('[%d]Expression Acc@1: %1.5f' % (global_step, acc1_meter.avg))
            logging.info('[%d]Expression Acc@1-Highest: %1.5f' % (global_step, self.highest_acc1))
            logging.info('[%d]Expression Acc@5: %1.5f' % (global_step, acc5_meter.avg))
            logging.info('[%d]Expression Acc@5-Highest: %1.5f' % (global_step, self.highest_acc5))
            logging.info('[%d]10 Times Expression Acc@1: %1.5f' % (global_step, self.limited_meter.avg))
            logging.info('[%d]10 Times Expression Acc@1-Highest: %1.5f' % (global_step, self.limited_meter.best))

    def __call__(self, num_update, model):
        # if self.rank is 0 and num_update > 0:
        model.eval()
        self.ver_test(model, num_update)
        model.train()
