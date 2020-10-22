import os
import gc
import time
import shutil
import random
import warnings
import typing as tp
from pathlib import Path
from contextlib import contextmanager

import yaml, json
from joblib import delayed, Parallel

import cv2
import librosa
import audioread
import soundfile as sf

import numpy as np
import pandas as pd
from audiomentations import Compose, AddGaussianNoise, TimeStretch

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions as ppe_extensions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from efficientnet_pytorch import EfficientNet
from torchvision import models, transforms

import model, utils

# pd.options.display.max_rows = 500
# pd.options.display.max_columns = 500

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    

@contextmanager
def timer(name: str) -> None:
    """Timer Util"""
    t0 = time.time()
    print("[{}] start".format(name))
    yield
    print("[{}] done in {:.0f} s".format(name, time.time() - t0))

ROOT = Path.cwd()
TRAIN_AUDIO_DIR = ROOT / "data" / "train_audio"
TRAIN_RESAMPLED_AUDIO_DIRS = [
  ROOT / "data" / "birdsong-resampled-train-audio-{:0>2}".format(i)  for i in range(5)
]
TEST_AUDIO_DIR = ROOT / "test_audio"

train = pd.read_csv(TRAIN_RESAMPLED_AUDIO_DIRS[0] / "train_mod.csv")

with open(ROOT / "configs" / "config.yaml", 'r') as yml:
    settings = yaml.load(yml)

with open(ROOT / "utils" / "birdcode.json", 'r') as jsn:
    BIRD_CODE = json.load(jsn)

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

PERIOD = 5

tmp_list = []
for audio_d in TRAIN_RESAMPLED_AUDIO_DIRS:
    if not audio_d.exists():
        continue
    for ebird_d in audio_d.iterdir():
        if ebird_d.is_file():
            continue
        for wav_f in ebird_d.iterdir():
            tmp_list.append([ebird_d.name, wav_f.name, wav_f.as_posix()])
            
train_wav_path_exist = pd.DataFrame(
    tmp_list, columns=["ebird_code", "resampled_filename", "file_path"])

del tmp_list

train_all = pd.merge(
    train, train_wav_path_exist, on=["ebird_code", "resampled_filename"], how="inner")

skf = StratifiedKFold(**settings["split"]["params"])

train_all["fold"] = -1
for fold_id, (train_index, val_index) in enumerate(skf.split(train_all, train_all["ebird_code"])):
    train_all.iloc[val_index, -1] = fold_id
    
# # check the propotion
fold_proportion = pd.pivot_table(train_all, index="ebird_code", columns="fold", values="xc_id", aggfunc=len)
print(fold_proportion.shape)

use_fold = settings["globals"]["use_fold"]
train_file_list = train_all.query("fold != @use_fold")[["file_path", "ebird_code"]].values.tolist()
val_file_list = train_all.query("fold == @use_fold")[["file_path", "ebird_code"]].values.tolist()

print("[fold {}] train: {}, val: {}".format(use_fold, len(train_file_list), len(val_file_list)))

set_seed(settings["globals"]["seed"])
device = torch.device(settings["globals"]["device"])
output_dir = Path(settings["globals"]["output_dir"])

# # # get loader
train_loader, val_loader = utils.dataloader.get_loaders_for_training(
    settings["dataset"]["params"], settings["loader"], train_file_list, val_file_list)

# # # get model
model = model.mymodel.get_model(settings["model"])
model = model.to(device)

# # # get optimizer
optimizer = getattr(
    torch.optim, settings["optimizer"]["name"]
)(model.parameters(), **settings["optimizer"]["params"])

# # # get scheduler
scheduler = getattr(
    torch.optim.lr_scheduler, settings["scheduler"]["name"]
)(optimizer, **settings["scheduler"]["params"])

# # # get loss
loss_func = getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])

# # # create training manager
trigger = None

manager = ppe.training.ExtensionsManager(
    model, optimizer, settings["globals"]["num_epochs"],
    iters_per_epoch=len(train_loader),
    stop_trigger=trigger,
    out_dir=output_dir
)

manager = utils.evaluation.set_extensions(
    manager, settings, model, device,
    val_loader, optimizer, loss_func,
)

utils.evaluation.train_loop(
    manager, settings, model, device,
    train_loader, optimizer, scheduler, loss_func
)
