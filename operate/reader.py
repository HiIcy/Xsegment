# coding:utf-8
# __author__ = deepSea
# __time__ = 2020/1/15
# __file__ = reader
# __desc__ =
from copy import deepcopy
from pathlib import Path
import random
import torch.cuda as cd
import numpy
import torch
from utils.metrics import AverageMeter
import os
import torch.nn as nn
import yaml
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import importlib
from data.dataloader import get_data_loader
from utils.metrics import Metrics
from torch.utils.tensorboard import SummaryWriter


class Processer(object):
    def __init__(self, args, crition: nn.CrossEntropyLoss, optimzer):
        seed = random.randint(0,1000)
        random.seed(seed)
        torch.manual_seed(seed)
        cd.manual_seed(seed)
        cd.manual_seed_all(seed)
        self.args = args
        self.model_name = args.net
        self.config = self._parse_args(args.config)
        net_module = importlib.import_module(f"net.{self.model_name}")
        self.model_class = getattr(net_module,self.model_name)
        self.model = self.model_class(**self._parse_model_args())
        self.crition = crition
        self.base_lr = self.config.get("lr", 0.01)

        self.optimizer = self._get_optimizer(optimzer)
        self.iters = self.config.get("iter", 5000)
        self.power = self.config.get("power", 0.9)
        self.numclass = self.config['numclass']
        self.batch_size = self.config['batch_size']
        self.print_freq = self.config['print_freq']
        self.save_freq = self.config['save_freq']
        self.gpu = self.config.get('gpus')
        print(f"gpus: {self.gpu}")
        if self.gpu:
            self.gpu = [self.gpu] if isinstance(self.gpu, int) else list(self.gpu)
        else:
            self.device = torch.device("cpu")
        self.train_dataloader = get_data_loader(
            self.config['train_data_path'],
            self.config['train_annot_path'],
            self.numclass,
            img_size=self.config['img_size'],
            batch_size=self.batch_size,
            name=self.config['dataset_name']
        )
        self.val_dataloader = get_data_loader(
            self.config['val_data_path'],
            self.config['val_annot_path'],
            self.numclass,
            img_size=self.config['img_size'],
            batch_size=self.batch_size,
            name=self.config['dataset_name'],
            mode='eval'
        )
        self.metricer = Metrics(self.numclass)
        logdir = self._get_log_dir()
        self.writer = SummaryWriter(log_dir=logdir)
        if self.gpu:
            print(torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=self.gpu).cuda(self.gpu[0])
            # self.crition = self.crition.cuda(self.gpu[0])
            cudnn.benchmark = False  # 加速1
            cudnn.deterministic = True
        # self.best_state_dict = deepcopy(self.model.state_dict())
        # self.best_acc = 0.
        # self.min_loss = numpy.inf

    def _get_optimizer(self,optimizer):
        modules_ori = [self.model.layer0, self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        modules_new = [self.model.ppm, self.model.fcn, self.model.aux_logits]
        modules_list = []
        for module in modules_ori:
            modules_list.append(dict(params=module.parameters(),lr=self.base_lr))
        for module in modules_new:
            # 底下几个学习率大点学的快些
            modules_list.append(dict(params=module.parameters(),lr=self.base_lr*10))
        return optimizer(modules_list,**self._parse_optimzer_args())

    def _get_log_dir(self):
        from datetime import datetime
        now = datetime.now()
        now = now.strftime("%Y_%m_%d")
        logdir_prefix = self.config['log_dir']
        logdir = Path(logdir_prefix) / now
        if not logdir.exists():
            logdir.mkdir()
        return str(logdir)

    def _parse_args(self, config):
        r = {}
        with open(config) as f:
            need_config = yaml.load(f,yaml.SafeLoader)
            if need_config:
                r = need_config
        return r

    def _parse_model_args(self):
        # r = {}
        d = self.config.get("model_param")
        # for v in d.values():
        #     r.append(v)
        return d

    def _parse_optimzer_args(self):
        r = self.config.get("optimizer_param", {})
        return r

    def adjust_lr(self, iter,index_split=5):
        lr = self.base_lr * ((1 - iter / self.iters) ** self.power)
        print(f"current learning-rate: {lr}")
        for index in range(0, index_split):
            self.optimizer.param_groups[index]['lr'] = lr
        for index in range(index_split, len(self.optimizer.param_groups)):
            self.optimizer.param_groups[index]['lr'] = lr * 10

    def single_train(self, iter):
        self.model.train()
        running_loss = AverageMeter()
        running_aux_loss = AverageMeter()
        running_mas_loss = AverageMeter()
        running_iou = AverageMeter()
        running_pixelacc = AverageMeter()  # 像素点的准确度
        count = 0
        # local_state_dict = deepcopy(self.best_state_dict)
        # pre_loss = numpy.inf
        print(f"{iter} / {self.iters}  ==== train  \n")
        for input in self.train_dataloader:
            data, label = input
            N,C,H,W = data.size()
            if torch.cuda.is_available() and self.gpu:
                data = data.cuda(self.gpu[0])
                label = label.cuda(self.gpu[0])
            label = label.long()
            self.optimizer.zero_grad()
            label = label.view(label.shape[0],*label.shape[2:])
            mas_o, aux_o = self.model(data)
            mas_o_m = mas_o.argmax(dim=1)
            self.metricer.loadData(mas_o_m.cpu().numpy(),
                                   label.cpu().numpy())

            # aux_o = aux_o.view(-1,self.numclass)
            # mas_o = mas_o.view(-1,self.numclass)
            # label = label.view(-1)

            aux_loss = self.crition(aux_o, label)
            mas_loss = self.crition(mas_o, label)
            mas_loss, aux_loss = torch.mean(mas_loss), torch.mean(aux_loss)
            aux_weight = self.config['aux_weight']
            loss = aux_loss * aux_weight + mas_loss
            # batch = N*H*W
            running_loss.update(loss.item(), N*H*W)
            running_aux_loss.update(aux_loss.item(), N*H*W)
            running_mas_loss.update(mas_loss.item(), N*H*W)
            running_pixelacc.update(self.metricer.pixelAccuracy())
            running_iou.update(self.metricer.meanIntersectionOverUnion())

            loss.backward()
            self.optimizer.step()
            if count % self.print_freq == self.print_freq - 1:
                print(f"[第{count//self.print_freq}次] --- aux_loss:{running_aux_loss.val}  "
                      f"mas_loss:{running_mas_loss.val}  "
                      f"loss:{running_loss.val}  "
                      f"pixelacc:{running_pixelacc.val}  "
                      f"meaniou:{running_iou.val}")

                self.writer.add_scalar("aux_loss", running_aux_loss.val,
                                       iter * len(self.train_dataloader) + count)
                self.writer.add_scalar("mas_loss", running_mas_loss.val,
                                       iter * len(self.train_dataloader) + count)
                self.writer.add_scalar("loss", running_loss.val,
                                       iter * len(self.train_dataloader) + count)

                self.writer.add_scalar("pixel_acc",running_pixelacc.val,
                                       iter*len(self.train_dataloader) + count)
                self.writer.add_scalar("mean_iou", running_iou.val,
                                       iter * len(self.train_dataloader) + count)

                # TODO:最小iou
                # if running_loss.val < self.min_loss and \
                #             running_pixelacc.val > self.best_acc:
                #     self.min_loss = running_loss.val
                #     self.best_acc = running_pixelacc.val
                #     local_state_dict = deepcopy(self.model.state_dict())
            torch.cuda.empty_cache()
            count += 1
            self.metricer.reset()

        # self.best_state_dict = local_state_dict
        print(f"train result at epoch [{iter}/{self.iters}] :mIou/mAcc: {running_iou.avg}/{running_pixelacc.avg}")

    def single_eval(self, iter):
        self.model.eval()
        running_mas_loss = AverageMeter()
        running_iou = AverageMeter()
        running_pixelacc = AverageMeter()  # 像素点的准确度
        count = 0
        print(f"{iter} / {self.iters}  ===   eval    \n")
        with torch.no_grad():
            for input in self.val_dataloader:
                data, label = input
                N,C,H,W = data.shape
                if torch.cuda.is_available() and self.gpu:
                    data = data.cuda(self.gpu[0])
                    label = label.cuda(self.gpu[0])
                label = label.long()
                label = label.view(label.shape[0],*label.shape[2:])
                mas_o = self.model(data)
                mas_o_m = mas_o.argmax(dim=1)

                self.metricer.loadData(mas_o_m.cpu().numpy(),
                                       label.cpu().numpy())

                mas_loss = self.crition(mas_o, label)
                running_mas_loss.update(mas_loss.item(),N*H*W)
                running_pixelacc.update(self.metricer.pixelAccuracy())
                running_iou.update(self.metricer.meanIntersectionOverUnion())

                if count % self.print_freq == self.print_freq - 1:
                    print(f"[{count//self.print_freq}次]   -----    mas_loss:{running_mas_loss.val}  "
                          f"pixelacc:{running_pixelacc.val}  "
                          f"meaniou:{running_iou.val}")

                    self.writer.add_scalar("eval_loss", running_mas_loss.val,
                                           iter * len(self.val_dataloader) + count)
                    self.writer.add_scalar("pixel_acc", running_pixelacc.val,
                                           iter * len(self.val_dataloader) + count)
                    self.writer.add_scalar("mean_iou", running_iou.val,
                                           iter * len(self.val_dataloader) + count)

                count += 1
                self.metricer.reset()

    def train(self):
        print("train length : ",len(self.train_dataloader.dataset))
        print("eval length : ",len(self.val_dataloader.dataset))
        try:
            for iter in range(self.iters):
                self.adjust_lr(iter)
                self.single_train(iter)
                self.single_eval(iter)
            self.model.train()
            self.save()
        except KeyboardInterrupt:
            self.save()

    def save(self, with_net=False):
        from datetime import datetime
        now = datetime.now()
        now = now.strftime("%Y_%m_%d")
        print(now)
        prefix = f"{self.model_name}_{now}_"
        if with_net:
            self.model.to("cpu")
            torch.save(self.model, str(Path(self.config['save_path'])/(prefix + "model.pth")))
        else:
            torch.save(self.model.module.state_dict(), str(Path(self.config['save_path'])/(prefix+"state_dict.pth")))
