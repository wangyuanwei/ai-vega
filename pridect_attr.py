import torch
import torchvision.transforms as transforms
from PIL import Image
from config_attr import Common


def get_label(score):
    label = []
    min_v = 1 / score.shape[0]
    sort_v = score.sort(descending=True)
    for idx in sort_v.indices.numpy():
        if score[idx] > min_v:
            label.append({"name": Common.labels[idx], "score": round(score[idx].detach().numpy().tolist(), 5)})
        else:
            break
    return label


def pridect(imagePath, modelPath):
    '''
    预测函数
    :param imagePath: 图片路径
    :param modelPath: 模型路径
    :return:
    '''
    # 1. 读取图片
    image = Image.open(imagePath)
    # 2. 进行缩放
    image = image.resize(Common.imageSize)
    image.show()
    # 3. 加载模型
    model = torch.load(modelPath)
    model = model.to(Common.device)
    # 4. 转为tensor张量
    transform = transforms.ToTensor()
    x = transform(image)
    x = torch.unsqueeze(x, 0)  # 升维
    x = x.to(Common.device)
    # 5. 传入模型
    output = model(x)
    print("预测结果：", get_label(output[0]))


if __name__ == '__main__':
    pic_path = "./test/nzk.jpeg"
    model_path = "./model/clothing_reg-attr_2023-02-28-16-37-02.pth"
    pridect(pic_path, model_path)
