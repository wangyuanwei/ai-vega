from flask import Flask, request
import torch
import torchvision.transforms as transforms
from PIL import Image
from config import Common

app = Flask(__name__)  # 模块名


@app.route("/")
def index():
    return "AI织女"


@app.route("/predict")
def predict():
    img_path = request.args.get('img_path', '')
    image = Image.open(img_path)
    # 进行缩放
    image = image.resize(Common.imageSize)
    # 转为tensor张量
    transform = transforms.ToTensor()
    x = transform(image)
    x = torch.unsqueeze(x, 0)  # 升维
    x = x.to(Common.device)
    # 5. 传入模型
    output = model(x)
    # 6. 使用argmax选出最有可能的结果
    output = torch.argmax(output)
    return Common.labels[output.item()]


if __name__ == "__main__":
    model_path = "./model/clothing_reg-2023-02-20-15-28-19.pth"
    model = torch.load(model_path)
    model = model.to(Common.device)
    app.run()  # 启动程序
