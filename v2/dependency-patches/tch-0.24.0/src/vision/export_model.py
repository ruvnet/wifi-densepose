# This script exports pre-trained model weights in the safetensors format.
import numpy as np
import torch
import torchvision
from safetensors import torch as stt

m = torchvision.models.efficientnet_b0(pretrained=True)
stt.save_file(m.state_dict(), 'efficientnet-b0.safetensors')
m = torchvision.models.efficientnet_b1(pretrained=True)
stt.save_file(m.state_dict(), 'efficientnet-b1.safetensors')
m = torchvision.models.efficientnet_b2(pretrained=True)
stt.save_file(m.state_dict(), 'efficientnet-b2.safetensors')
m = torchvision.models.efficientnet_b3(pretrained=True)
stt.save_file(m.state_dict(), 'efficientnet-b3.safetensors')
m = torchvision.models.efficientnet_b4(pretrained=True)
stt.save_file(m.state_dict(), 'efficientnet-b4.safetensors')
