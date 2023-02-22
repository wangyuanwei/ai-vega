import torch
import torchvision.transforms as transforms
from PIL import Image
from config import Common


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
    # 6. 使用argmax选出最有可能的结果
    output = torch.argmax(output)
    print("预测结果：", Common.labels[output.item()])


if __name__ == '__main__':
    pic_path = "/Users/wyw/Desktop/test/cate=打底裤,title=蚕莎芭比外穿打底瑜伽提臀收腹春秋鲨鱼骑行长裤路薄款沙女神夏季,attr=长裤,.jpeg"
    model_path = "./model/clothing_reg-2023-02-20-15-28-19.pth"
    pridect(pic_path, model_path)
