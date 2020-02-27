# coding:utf-8
# __author__ = deepSea
# __time__ = 2020/1/18
# __file__ = evaler
# __desc__ =
import importlib
import torch
import yaml
from torch import nn
from utils.metrics import Metrics,AverageMeter
from data import get_data_loader
from utils.visualization import draw_color,colorize
import cv2
from pathlib import Path
import numpy as np
import torch.nn.functional as F

class Evaler:
    def __init__(self,args,crition: nn.CrossEntropyLoss):
        self.args = args
        self.model_name = args.net
        self.config = self._parse_args(args.config)
        net_module = importlib.import_module(f"net.{self.model_name}")
        self.model_class = getattr(net_module, self.model_name)
        self.model:torch.nn.Module = self.model_class(*self._parse_model_args())
        self.numclass = self.config['numclass']
        self.save_path = self.config['save_path']
        self.batch_size = self.config['batch_size']
        self.crition = crition
        self.metricer = Metrics(self.numclass)
        self.test_dataloader = get_data_loader(
            self.config['test_data_path'],
            self.config['test_annot_path'],
            self.numclass,
            img_size=self.config['ori_size'],
            batch_size=8,
            name=self.config['dataset_name'],
            mode='test',
            return_name=True,
        )
        self.model.load_state_dict(torch.load(self.save_path,
                                              map_location=torch.device("cuda:0")),strict=False)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
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
    def net_process(self, image, mean, std=None, flip=False):
        input = torch.from_numpy(image.transpose((2, 0, 1))).float()
        if std is None:
            for t, m in zip(input, mean):
                t.sub_(m)
        else:
            for t, m, s in zip(input, mean, std):
                t.sub_(m).div_(s)
        input = input.unsqueeze(0).cuda()
        if flip:
            input = torch.cat([input, input.flip(3)], 0)
        with torch.no_grad():
            output = self.model(input)
        _, _, h_i, w_i = input.shape
        _, _, h_o, w_o = output.shape
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        if flip:
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        output = output.data.cpu().numpy()
        output = output.transpose(1, 2, 0)
        return output

    def scale_process(self, image, crop_h, crop_w, h, w, mean, std=None, stride_rate=2 / 3):
        ori_h, ori_w, _ = image.shape
        # 填充周边
        pad_h = max(crop_h - ori_h, 0)
        pad_w = max(crop_w - ori_w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=mean)
        new_h, new_w, _ = image.shape
        # FAQ:
        stride_h = int(np.ceil(crop_h * stride_rate))
        stride_w = int(np.ceil(crop_w * stride_rate))
        grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
        grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
        prediction_crop = np.zeros((new_h, new_w, self.numclass), dtype=float)
        count_crop = np.zeros((new_h, new_w), float)
        for index_h in range(0, grid_h):
            for index_w in range(0, grid_w):
                s_h = index_h * stride_h
                e_h = min(s_h + crop_h, new_h)
                s_h = e_h - crop_h
                s_w = index_w * stride_w
                e_w = min(s_w + crop_w, new_w)
                s_w = e_w - crop_w
                image_crop = image[s_h:e_h, s_w:e_w].copy()
                count_crop[s_h:e_h, s_w:e_w] += 1
                prediction_crop[s_h:e_h, s_w:e_w, :] += self.net_process(image_crop, mean, std)
        prediction_crop /= np.expand_dims(count_crop, 2)
        prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
        prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return prediction
    # copy other's code,not yet understand
    def __fest(self,visual=False):
        cpth = "/data/soft/javad/Xsegment/data/camvid/cityscapes_colors.txt"
        colors = np.loadtxt(cpth).astype('uint8')
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        test_h, test_w = self.config['img_size']
        base_size = self.config['base_size']
        scales = self.config['scales']
        self.model.eval()
        batch_time = AverageMeter()

        running_mas_loss = AverageMeter()
        running_iou = AverageMeter()
        running_pixelacc = AverageMeter()  # 像素点的准确度
        r = []
        result_path = self.config['result_path']
        if not Path(result_path).exists():
            Path(result_path).mkdir()
        for i, input in enumerate(self.test_dataloader):
            input = input[0]
            input = np.squeeze(input.numpy(), axis=0)
            image = np.transpose(input, (1, 2, 0))
            h, w, _ = image.shape
            prediction = np.zeros((h, w, self.numclass), dtype=float)
            for scale in scales:
                long_size = round(scale * base_size)
                new_h = long_size
                new_w = long_size
                if h > w:
                    new_w = round(long_size / float(h) * w)
                else:
                    new_h = round(long_size / float(w) * h)
                image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                prediction += self.scale_process(image_scale, test_h, test_w, h, w, mean, std)
            prediction /= len(scales)
            prediction = np.argmax(prediction, axis=2)
            # end = time.time()
            gray = np.uint8(prediction)
            # color = colorize(gray, colors)
            segim = draw_color(prediction, self.numclass)
            cv2.imwrite(f"{result_path}/visual_{i}_.jpg", segim)
            # color.save(f"{result_path}/visual_{i}_.png")
    def test(self,visual=True):
        
        test_h, test_w = self.config['img_size']
        ori_h,ori_w = self.config['ori_size']
        self.model.eval()
        batch_time = AverageMeter()

        running_mas_loss = AverageMeter()
        running_iou = AverageMeter()
        running_pixelacc = AverageMeter()  # 像素点的准确度
        r = []
        ie = []
        result_path = self.config['result_path']
        if not Path(result_path).exists():
            Path(result_path).mkdir()
        with torch.no_grad():
            for i, input in enumerate(self.test_dataloader):
                data, label,img_name = input
                N,_,H,W = data.shape
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                label = label.long()
                label = label.view(label.shape[0], *label.shape[2:])
                mas_o = self.model(data)
                mas_o_m = mas_o.argmax(dim=1)
                # print((mas_o_m==0).sum().item())
                for m,imn in zip(mas_o_m,img_name):
                    m = m.cpu().numpy().astype(np.uint8)
                    r.append(m)
                    ie.append(imn)
                self.metricer.loadData(mas_o_m.cpu().numpy(),
                                       label.cpu().numpy())
                # mas_loss = self.crition(mas_o, label)
                # running_mas_loss.update(mas_loss.item(),N*H*W)
                running_pixelacc.update(self.metricer.pixelAccuracy())
                running_iou.update(self.metricer.meanIntersectionOverUnion())
                torch.cuda.empty_cache()

                self.metricer.reset()
        print(f"test :  "
              f"pixelacc:{running_pixelacc.avg:.5f}  " 
              f"meaniou:{running_iou.avg:.5f}")
        if visual:
            for pred,imn in zip(r,ie):
                segim = draw_color(pred,self.numclass)
                cv2.imwrite(f"{result_path}/{imn}.jpg",segim)





