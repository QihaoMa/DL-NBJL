# coding=gbk
import time
from PIL import Image
import torch
import torch.onnx

import torchvision.transforms as transforms
from models.ENet.enet import ENet



def convertPT(model_path):
    device = torch.device('cpu')
    net = ENet(num_classes=3,channels=[8,32,64],factor=1,sp_layer_mid=False,sp_layer_bottle=True).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device),strict=False)
    net.eval().cpu()
    input = torch.rand(1, 3, 640, 480)
    traced_script_module = torch.jit.trace(func=net, example_inputs=input)
    nowTime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    traced_script_module.save('./PT/SPENet_{}.pt'.format(nowTime))
    print('finish')


def convertONNX(model_path):
    device = torch.device('cpu')
    net = ENet(num_classes=3)
    net.load_state_dict(torch.load(model_path,map_location=device))
    net.eval()
    x = torch.randn(1,3,560,168).to(device)
    torch.onnx.export(net,x,'./ONNX/ENet_560_168.onnx',export_params=True,verbose=True,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

