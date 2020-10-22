import typing as tp
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def get_model(args: tp.Dict):
    model = EfficientNet.from_pretrained(args['name'])
    model._conv_stem = nn.Sequential(nn.Conv2d(4, 32, (3, 3), (2, 2), bias=False), nn.ZeroPad2d(padding=(1, 1, 1, 1)))
    model._fc = nn.Sequential(
        nn.BatchNorm1d(1280),
        nn.Linear(1280, 528), nn.BatchNorm1d(528), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(528, 264))
    # checkpoint = torch.load(weights_path)
    # model.load_state_dict(checkpoint)
    
    return model
