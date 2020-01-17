# coding:utf-8
# __author__ = hiicy redldw
# __time__ = 2020/1/14
# __file__ = dataloader
# __desc__ =
from torchvision import transforms
from torch.utils.data import DataLoader
import os


def get_data_loader(img_path, anno_path, numclass, img_size=(400, 480),
                            batch_size=8, name="camvid",mode="train"):
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
    dataset = segDataset(img_path, anno_path, _getTransforms(mode=mode),
                         numclass, img_size)
    cpu_num = os.cpu_count()
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True,
                      drop_last=True,
                      num_workers=0)  # TODO:collec fn


def _getTransforms(degree=(-10, 10), mode="train"):
    return [
        # 随机翻转
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degree),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ] if mode == "train" else [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
