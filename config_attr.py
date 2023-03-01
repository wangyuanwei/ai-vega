import time

import torch


# 项目配置文件

class Common:
    '''
    通用配置
    '''
    basePath = "/Users/wyw/data/vega/picture/"  # 图片文件基本路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备配置
    imageSize = (224, 224)  # 图片大小
    labels = [
        "宽松",
        "高腰",
        "加绒",
        "显瘦",
        "加厚",
        "黑色",
        "长袖",
        "休闲",
        "薄款",
        "直筒",
        "短袖",
        "白色",
        "韩版",
        "紧身",
        "高领",
        "修身",
        "ins",
        "纯棉",
        "大码",
        "九分",
        "阔腿",
        "短裤",
        "垂感",
        "灰色",
        "针织",
        "纯色",
        "半袖",
        "长裤",
        "拖地",
        "束脚",
        "圆领",
        "中长款",
        "冰丝",
        "连帽",
        "羊羔绒",
        "无袖",
        "德绒",
        "螺纹",
        "开叉",
        "蕾丝",
        "印花",
        "金丝绒",
        "灯芯绒",
        "摇粒绒",
        "法式",
        "长款",
        "肉色",
        "条纹",
        "格子",
        "雪纺",
        "简约",
        "字母",
        "翻领",
        "八分",
        "七分",
        "五分裤",
        "中领",
        "蓝色",
        "浅色",
        "红色",
    ]
    label_num = len(labels)


class Train:
    '''
    训练相关配置
    '''
    batch_size = 128
    num_workers = 0  # 对于Windows用户，这里应设置为0，否则会出现多线程错误
    lr = 0.001
    epochs = 5
    logDir = "./log/" + time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())  # 日志存放位置
    modelDir = "./model/"  # 模型存放位置
