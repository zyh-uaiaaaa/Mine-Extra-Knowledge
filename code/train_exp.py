import argparse
import logging
import os

import math
import torch.nn as nn
from torch import distributed
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from backbones import get_model
from lr_scheduler import build_scheduler

from expression import *

from utils.utils_callbacks import CallBackLogging
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

    
def ACLoss(att_map1, att_map2, grid_l, output):
    flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
    flip_grid_large = Variable(flip_grid_large, requires_grad = False)
    flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
    att_map2_flip = F.grid_sample(att_map2, flip_grid_large, mode = 'bilinear', padding_mode = 'border', align_corners=True)
    flip_loss_l = F.mse_loss(att_map1, att_map2_flip, reduction='none')
    return flip_loss_l    


def generate_flip_grid(w, h):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid

class LSR2(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e

    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        mask = (one_hot==0)
        balance_weight = torch.tensor([0.95124031, 4.36690391, 1.71143654, 0.25714585, 0.6191221, 1.74056738, 0.48617274]).to(one_hot.device)
        ex_weight = balance_weight.expand(one_hot.size(0),-1)
        resize_weight = ex_weight[mask].view(one_hot.size(0),-1)
        resize_weight /= resize_weight.sum(dim=1, keepdim=True)
        one_hot[mask] += (resize_weight*smooth_factor).view(-1)
        
#         one_hot += smooth_factor / length
        return one_hot.to(target.device)

    def forward(self, x, target):
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)
        return torch.mean(loss)
    

def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    # Dataloader
    expression_train_loader = get_analysis_train_dataloader("Expression", cfg)

    # Backbone
    swin = get_model(cfg.network).cuda()

    model = SwinTransFER(swin=swin, swin_num_features=768, num_classes=7, cam=True).cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    model.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    #model._set_static_graph()

    cfg.total_batch_size = world_size * cfg.batch_size
    cfg.epoch_step = len(expression_train_loader)
    cfg.num_epoch = math.ceil(cfg.total_step / cfg.epoch_step)

    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    # Loss
    expression_criteria = torch.nn.CrossEntropyLoss()

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=[{"params": model.module.parameters(), 'lr': cfg.lr}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=[{"params": model.module.parameters(), 'lr': cfg.lr}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    lr_scheduler = build_scheduler(
        optimizer=opt,
        lr_name=cfg.lr_name,
        warmup_lr=cfg.warmup_lr,
        min_lr=cfg.min_lr,
        num_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step)

    start_epoch = 0
    global_step = 0

    if cfg.init:
#         dict_checkpoint = torch.load(os.path.join(cfg.init_model, f"start_{rank}.pt"))
        dict_checkpoint = torch.load(os.path.join(cfg.init_model, f"start_0.pt"))
        model.module.encoder.load_state_dict(dict_checkpoint["state_dict_backbone"], strict=True)  # only load backbone!
        del dict_checkpoint

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_step_{cfg.resume_step}_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        local_step = dict_checkpoint["local_step"]

        if local_step == cfg.epoch_step - 1:
            start_epoch = start_epoch+1
            local_step = 0
        else:
            local_step += 1

        global_step += 1

        model.module.load_state_dict(dict_checkpoint["state_dict_model"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    expression_val_dataloader = get_analysis_val_dataloader("Expression", config=cfg)
    expression_verification = ExpressionVerification(data_loader=expression_val_dataloader, summary_writer=summary_writer)

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)


    
    #with torch.no_grad():
    #    expression_verification(global_step, model)
    

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(expression_train_loader, DataLoader):
            expression_train_loader.sampler.set_epoch(epoch)

        for idx, data in enumerate(expression_train_loader):

            # skip
            if cfg.resume:
                if idx < local_step:
                    continue

            expression_img, expression_label, expression_img1 = data
            expression_img = expression_img.cuda(non_blocking=True)
            expression_label = expression_label.cuda(non_blocking=True)
            expression_img1 = expression_img1.cuda(non_blocking=True)

            expression_output, hm = model(expression_img)
            expression_output1, hm1 = model(expression_img1)            
#             ########## added
            grid_l = generate_flip_grid(7, 7).cuda(non_blocking=True)  
            flip_loss = ACLoss(hm, hm1, grid_l, expression_output)    #N*7*7*7
            
        
            flip_loss = flip_loss.mean(dim=-1).mean(dim=-1) #N*7
            balance_weight = torch.tensor([0.95124031, 4.36690391, 1.71143654, 0.25714585, 0.6191221, 1.74056738, 0.48617274]).cuda().view(7,1)
            flip_loss = torch.mm(flip_loss, balance_weight).squeeze()
            
            loss = LSR2(0.3)(expression_output, expression_label) + 0.1 * flip_loss.mean()



            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step_update(global_step)

            with torch.no_grad():
                loss_am.update(loss.item(), 1)

                callback_logging(global_step, loss_am, epoch, cfg.fp16,
                                 lr_scheduler.get_update_values(global_step)[0], amp)

                if (global_step+1) % cfg.verbose == 0:
                    expression_verification(global_step, model)

            if cfg.save_all_states and (global_step+1) % cfg.save_verbose == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "local_step": idx,
                    "state_dict_model": model.module.state_dict(),
                    "state_optimizer": opt.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(cfg.output, f"copy2_checkpoint_step_{global_step}_gpu_{rank}.pt"))

            # update
            if global_step >= cfg.total_step - 1:
                break  # end
            else:
                global_step += 1

        if global_step >= cfg.total_step - 1:
            break
        if cfg.dali:
            expression_train_loader.reset()

    with torch.no_grad():
        expression_verification(global_step, model)

    distributed.destroy_process_group()



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
