import torch
import torchvision.transforms as transforms
from PIL import Image
from model.DelicacyNet import DelicacyNet
import collections
from model.EFEBlock import build_backbone
from model.EncoderToDecoder import build_EncoderToDecoder
import pandas as pd

def test_single(args):
    # 加载模型
    state_dict = torch.load('./output/checkpoint.pth')
    # 提取模型参数
    state_dict = state_dict['model']

    # 创建新的模型对象
    backbone = build_backbone(args)

    EncoderToDecoder = build_EncoderToDecoder(args)

    model = DelicacyNet(
        backbone,
        EncoderToDecoder,
        num_classes=2000,
        dim=512
    )
    model.load_state_dict(state_dict)
    model.to(args.device)

    # 预处理图像
    normalize = transforms.Normalize(mean=[0.5457954, 0.44430383, 0.34424934],
                                     std=[0.23273608, 0.24383051, 0.24237761])

    transform = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # 加载图像
    image = Image.open('160.jpg')

    # 预处理图像
    image = transform(image)
    image = image.to(args.device)

    # 将图像输入模型并运行推理
    model.eval()
    with torch.no_grad():
        output = model.forward(image.unsqueeze(0))

    # 获取分类结果
    pred = output['pred_logits']
    pred = torch.argmax(pred, dim=2)
    pred = int(pred)
    m = pd.read_csv('map.csv')
    print(pred)
    print("Protein(%): {}".format(m.loc[pred,'protein']))
    print("carbonhydrate(%): {}".format(m.loc[pred,'carbonhydrate']))
    print("fat(%): {}".format(m.loc[pred,'fat']))
    print("minerals(%): {}".format(m.loc[pred,'minerals']))
    print("fibre(%): {}".format(m.loc[pred,'fibre']))
    print("water(%): {}".format(m.loc[pred,'water']))
