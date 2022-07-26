from models.hyper_prior_compressor import HyperPrior
import torch
import argparse
from utils import load_checkpoint
from PIL import Image
from torchvision import transforms
import json
from env import AttrDict, build_env
import os

from skimage.metrics import peak_signal_noise_ratio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='HyperPrior', type=str)
parser.add_argument('--training_dir', default=r'E:\dataset\vimoe\train', type=str)
parser.add_argument('--validation_dir', default=r'E:\dataset\vimoe\test', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoint', type=str)
parser.add_argument('--config_file', default=r'E:\Git_repos\Hyper-Prior-Byf\configs\config.json', type=str)
parser.add_argument('--training_epochs', default=3000, type=int)
parser.add_argument('--stdout_interval', default=5, type=int)
parser.add_argument('--checkpoint_interval', default=5000, type=int)
parser.add_argument('--summary_interval', default=100, type=int)
parser.add_argument('--validation_interval', default=1000, type=int)
parser.add_argument('--fine_tuning', default=False, type=bool)
parser.add_argument('--Lambda', default=0.0067, type=float)

a = parser.parse_args()

with open(a.config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)
build_env(a.config_file, 'config.json', os.path.join(a.checkpoint_path, a.model_name))

device = torch.device('cuda:0')
compressor = HyperPrior(a, h, 0, 128, 192)
state_dict_com = load_checkpoint(r"checkpoint/HyperPrior/0130", device)
compressor.load_state_dict(state_dict_com['compressor'])

image = Image.open(r"E:\dataset\KoDak\kodim04.png").convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor()
])
img = transform(image)
img = img.unsqueeze(0).cuda()
compressor = compressor.to(device)

inv_transform = transforms.ToPILImage()

loss, bpp, bpp_y, bpp_z, distortion, img_reco = compressor(img)
img_reco = inv_transform(img_reco[0])
img_reco.save(r"C:\Users\EsakaK\Desktop\res.png")
print("loss:{}  bpp:{}  bpp_y:{}  bpp_z:{}  distortion:{}".format(loss, bpp, bpp_y, bpp_z, distortion))

# psnr = peak_signal_noise_ratio(np.asarray(image), np.asarray(img_reco))
# print("psnr is {:.4f}".format(psnr))
# print("bpp is {:.4f}".format(bpp))
