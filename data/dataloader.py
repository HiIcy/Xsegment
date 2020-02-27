# coding:utf-8
# __author__ = hiicy redldw
# __time__ = 2020/1/14
# __file__ = dataloader
# __desc__ =
from torchvision import transforms
from torch.utils.data import DataLoader
from .import dataaugment
import os


def get_data_loader(img_path, anno_path, numclass, img_size=(360, 480),
                            batch_size=8, name="camvid",mode="train",return_name=False):
    def _load_dataset(name):
        name = name.lower()
        prefix = f"data.{name}"
        import importlib
        try:
            module = importlib.import_module(prefix)
            dataset = getattr(module, f"{name.title()}Set")
        except:
            print(f"not find {name} class")
            exit(-1)
        else:
            return dataset

    segDataset = _load_dataset(name)
    dataset = segDataset(img_path, anno_path, __getTransforms(imgsize=img_size,mode=mode),
                         numclass,return_name)
    cpu_num = os.cpu_count()
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True,
                      drop_last=True,
                      pin_memory=True,
                      num_workers=0)  # TODO:collec fn


def __getTransforms(degree=(-10,10),scale=(0.5,2.),imgsize=(360,480),mode="train"):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    if mode=='train':
        dtfm = dataaugment.Compose([
            dataaugment.RandScale([scale[0], scale[1]]),
            dataaugment.RandRotate([degree[0], degree[1]], padding=mean, ignore_label=255),
            dataaugment.RandomGaussianBlur(),
            dataaugment.RandomHorizontalFlip(),
            dataaugment.Crop([imgsize[0],imgsize[1]], crop_type='rand', padding=mean, ignore_label=255),
            dataaugment.ToTensor(),
            dataaugment.Normalize(mean=mean, std=std)]
        )
    elif mode == 'eval':
        dtfm = dataaugment.Compose([
            dataaugment.Crop([imgsize[0],imgsize[1]], crop_type='center', padding=mean,
                           ignore_label=255),
            dataaugment.ToTensor(),
            dataaugment.Normalize(mean=mean, std=std)]
        )
    else:
        dtfm = dataaugment.Compose([
            dataaugment.Crop([imgsize[0], imgsize[1]], crop_type='center', padding=mean,
                             ignore_label=255),
            dataaugment.ToTensor(),
            dataaugment.Normalize(mean=mean, std=std)])
    return dtfm

def _getTransforms(degree=(-10, 10), mode="train"):
    return [
        # 随机翻转
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degree),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ] if mode == "train" else [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
