# coding:utf-8
# __author__ = deepSea
# __time__ = 2020/1/15
# __file__ = reader
# __desc__ =
from copy import deepcopy
from pathlib import Path
import numpy
import torch
import torch.nn as nn
import yaml
import importlib
from data.dataloader import get_data_loader
from utils.metrics import Metrics
from torch.utils.tensorboard import SummaryWriter


class Processer(object):
    def __init__(self, args, crition: nn.CrossEntropyLoss, optimzer):
        self.args = args
        self.model_name = args.net
        self.config = self._parse_args(args.config)
        net_module = importlib.import_module(f"net.{self.model_name}")
        self.model_class = getattr(net_module,self.model_name)
        self.model = self.model_class(*self._parse_model_args())
        self.crition = crition
        self.optimizer = optimzer(self.model.parameters(),
                                  **self._parse_optimzer_args())
        self.base_lr = self.config.get("lr", 0.01)
        self.iters = self.config.get("iter", 5000)
        self.power = self.config.get("power", 0.9)
        self.numclass = self.config['numclass']
        self.batch_size = self.config['batch_size']
        self.print_freq = self.config['print_freq']
        self.save_freq = self.config['save_freq']
        self.gpu = self.config.get('gpus')
        if self.gpu:
            self.gpu = [self.gpu] if isinstance(self.gpu, int) else list(self.gpu)
            self.device = torch.device(f"cuda:{self.gpu[0]}")
        else:
            self.device = torch.device("cpu")
        self.train_dataloader = get_data_loader(
            self.config['train_data_path'],
            self.config['train_annot_path'],
            self.numclass,
            batch_size=self.batch_size,
            name=self.config['dataset_name']
        )
        self.val_dataloader = get_data_loader(
            self.config['val_data_path'],
            self.config['val_annot_path'],
            self.numclass,
            batch_size=self.batch_size,
            name=self.config['dataset_name']
        )
        self.metricer = Metrics(self.numclass)
        logdir = self._get_log_dir()
        self.writer = SummaryWriter(log_dir=logdir)
        if self.gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.gpu).cuda()
        self.best_state_dict = deepcopy(self.model.state_dict())
        self.best_acc = 0.
        self.min_loss = numpy.inf

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
        r = []
        d = self.config.get("model_param")
        for v in d.values():
            r.append(v)
        return r

    def _parse_optimzer_args(self):
        r = self.config.get("optimizer_param", {})
        return r

    def adjust_lr(self, iter):
        lr = self.base_lr * ((1 - iter / self.iters) ** self.power)
        print(f"current learning-rate: {lr}")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def single_train(self, iter):
        self.model.train()
        running_loss = 0.
        running_aux_loss = 0.
        running_mas_loss = 0.
        running_iou = 0.
        running_pixelacc = 0.  # 像素点的准确度
        count = 0
        local_state_dict = deepcopy(self.best_state_dict)
        # pre_loss = numpy.inf
        print(f"{iter} / {self.iters}  ==== train  \n")
        for input in self.train_dataloader:
            data, label = input
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            label = label.long()
            label = label.view(label.shape[0],*label.shape[2:])
            mas_o, aux_o = self.model(data)
            mas_o_m = mas_o.argmax(dim=1)
            self.metricer.loadData(mas_o_m.cpu().numpy(),
                                   label.cpu().numpy())

            aux_loss = self.crition(aux_o, label)
            mas_loss = self.crition(mas_o, label)
            aux_weight = self.config['aux_weight']
            loss = aux_loss * aux_weight + (1 - aux_weight) * mas_loss

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_aux_loss += aux_loss.item()
            running_mas_loss += mas_loss.item()
            running_pixelacc += self.metricer.pixelAccuracy()
            running_iou += self.metricer.meanIntersectionOverUnion()

            if count % self.print_freq == self.print_freq - 1:
                print(f"[第{count//self.print_freq}次] --- aux_loss:{running_aux_loss / self.print_freq:.3f}  "
                      f"mas_loss:{running_mas_loss / self.print_freq:.3f}  "
                      f"loss:{running_loss / self.print_freq:.3f}  "
                      f"pixelacc:{running_pixelacc / self.print_freq:.3f}  "
                      f"meaniou:{running_iou/self.print_freq:.3f}")

                self.writer.add_scalar("aux_loss", running_aux_loss / self.print_freq,
                                       iter * len(self.train_dataloader) + count)
                self.writer.add_scalar("mas_loss", running_mas_loss / self.print_freq,
                                       iter * len(self.train_dataloader) + count)
                self.writer.add_scalar("loss", running_loss / self.print_freq,
                                       iter * len(self.train_dataloader) + count)

                self.writer.add_scalar("pixel_acc",running_pixelacc / self.print_freq,
                                       iter*len(self.train_dataloader) + count)
                self.writer.add_scalar("mean_iou", running_iou / self.print_freq,
                                       iter * len(self.train_dataloader) + count)

                # TODO:最小iou
                if running_loss / self.print_freq < self.min_loss and \
                            running_pixelacc / self.print_freq > self.best_acc:
                    self.min_loss = running_loss / self.print_freq
                    self.best_acc = running_pixelacc
                    local_state_dict = deepcopy(self.model.state_dict())

                running_loss = 0.
                running_aux_loss = 0.
                running_mas_loss = 0.
                running_iou = 0.
                running_pixelacc = 0.
            count += 1
            self.metricer.reset()

        self.best_state_dict = local_state_dict

    def single_eval(self, iter):
        self.model.eval()
        running_mas_loss = 0.
        running_iou = 0.
        running_pixelacc = 0.  # 像素点的准确度
        count = 0
        print(f"{iter} / {self.iters}  ===   eval    \n")
        with torch.no_grad():
            for input in self.val_dataloader:
                data, label = input
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                label = label.long()
                label = label.view(label.shape[0],*label.shape[2:])
                mas_o = self.model(data)
                mas_o_m = mas_o.argmax(dim=1)

                self.metricer.loadData(mas_o_m.cpu().numpy(),
                                       label.cpu().numpy())
                mas_loss = self.crition(mas_o, label)
                running_mas_loss += mas_loss.item()
                running_pixelacc += self.metricer.pixelAccuracy()
                running_iou += self.metricer.meanIntersectionOverUnion()

                if count % self.print_freq == self.print_freq - 1:
                    print(f"[{count//self.print_freq}次]   -----    mas_loss:{running_mas_loss / self.print_freq:.3f}  "
                          f"pixelacc:{running_pixelacc / self.print_freq:.3f}  "
                          f"meaniou:{running_iou/self.print_freq:.3f}")

                    self.writer.add_scalar("eval_loss", running_mas_loss / self.print_freq,
                                           iter * len(self.val_dataloader) + count)
                    self.writer.add_scalar("pixel_acc", running_pixelacc / self.print_freq,
                                           iter * len(self.val_dataloader) + count)
                    self.writer.add_scalar("mean_iou", running_iou / self.print_freq,
                                           iter * len(self.val_dataloader) + count)

                count += 1
                self.metricer.reset()

    def train(self):
        print("train length : ",len(self.train_dataloader.dataset))
        print("eval length : ",len(self.val_dataloader.dataset))
        for iter in range(self.iters):
            try:
                self.adjust_lr(iter)
                self.single_train(iter)
                self.single_eval(iter)
            except KeyboardInterrupt:
                self.save()
        self.save()

    def save(self, with_net=False):
        from datetime import datetime
        self.model.load_state_dict(self.best_state_dict)
        now = datetime.now()
        now = now.strftime("%Y_%m_%D")
        prefix = f"{self.model_name}_{now}_{self.min_loss}_"
        if with_net:
            self.model.to("cpu")
            torch.save(self.model, str(Path(self.config['save_path'])/(prefix + "model.pkl")))
        else:
            torch.save(self.model.state_dict(), str(Path(self.config['save_path'])/(prefix+"state_dict.pkl")))
