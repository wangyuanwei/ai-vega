from flask import Flask, request
import torch
import torchvision.transforms as transforms
from PIL import Image
from config import Common
from config_attr import Common as Common_attr
import json

app = Flask(__name__)  # 模块名


def get_attr_label(score):
    label = []
    min_v = 1 / score.shape[0]
    sort_v = score.sort(descending=True)
    for idx in sort_v.indices.numpy():
        if score[idx] > min_v:
            label.append({"name": Common_attr.labels[idx], "score": round(score[idx].detach().numpy().tolist(), 5)})
        else:
            break
    return label


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

    # 类目预测
    output_cate = model_cate(x)
    output_max_idx = torch.argmax(output_cate)
    cate_pre = [
        {"name": Common.labels[output_max_idx.item()],
         "score": round(output_cate[0][output_max_idx].detach().numpy().tolist(), 5)},
    ]

    # 属性预测
    output_attr = model_attr(x)
    attr_pre = get_attr_label(output_attr[0])

    res = {"data": {"category": cate_pre, "attribute": attr_pre}}

    return json.dumps(res, ensure_ascii=False)


if __name__ == "__main__":
    # 类目预测
    model_cate_path = "./model/clothing_reg-2023-02-20-15-28-19.pth"
    model_cate = torch.load(model_cate_path)
    model_cate = model_cate.to(Common.device)
    # 属性预测
    model_attr_path = "./model/clothing_reg-attr_2023-02-28-16-37-02.pth"
    model_attr = torch.load(model_attr_path)
    model_attr = model_attr.to(Common.device)

    app.run()  # 启动程序
