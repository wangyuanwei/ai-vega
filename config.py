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
        "T恤",
        "休闲裤",
        "打底裤",
        "牛仔裤",
        "毛针织衫",
        "卫衣|绒衫",
        "背心吊带",
        "半身裙",
        "衬衫",
        "毛衣",
        "短外套",
        "中老年女装",
        "连衣裙",
        "羽绒服",
        "西装",
        "棉衣|棉服",
        "大码女装",
        "马夹",
        "时尚套装",
        "休闲运动套装",
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
