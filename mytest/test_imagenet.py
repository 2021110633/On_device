# 导入所需的库
import logging
import torch
import data
import torchvision
import torchmetrics
import time
import csv
import openpyxl
import numpy as np
import argparse

# 从models目录导入模型相关的模块
from models import imagebind_model  # ImageBind 原始模型
# from models import lumen_model  # 自定义模型 lumen_model
# from models import lumen6_model  # 自定义模型 lumen6_model
from models.imagebind_model import ModalityType, load_module  # 导入核心类和函数
# from models.lumen6_model_test import ModalityType, load_module  # 从另一个测试文件导入
# from models.lumen_model import ModalityType, load_module  # 从lumen模型导入
# from models import lora_lumen as LoRA  # 导入LoRA微调模块

# 导入用于处理数据集的库
from pycocotools.coco import COCO
from torchvision import transforms  # 图像转换库
from torch.utils.data import DataLoader  # 数据加载器
from torchvision.datasets import CocoDetection  # COCO数据集类

# 导入更多数据集和度量库
from torchvision.datasets import ImageNet
from api.imagenet import ImageNetDataset  # 自定义ImageNet数据集类
from api.coco import CoCoDataset  # 自定义COCO数据集类
from metrics.accuracy import Accuracy  # 自定义准确率计算类

# 配置日志，用于打印信息
logging.basicConfig(level=logging.INFO, force=True)

print("test_imagebet")

# ==================== 配置参数 ====================
lora = False # 是否使用LoRA微调
linear_probing = False  # 是否使用线性探测，与LoRA互斥
load_head_post_proc_finetuned = False  # 是否加载微调后的头部和后处理模块

# 数据集路径配置
imagenet_datadir = "../.datasets/imagenet"
coco_datadir = "../.datasets/coco/val2017"
coco_annotation_file = '../.datasets/coco/annotations/instances_val2017.json'
lora_dir = '../data/air/pc/Mobile-Search-Engine/.checkpoints/lora31-1'  # LoRA检查点路径

# ==================== 命令行参数解析 ====================
# 使用argparse创建命令行参数解析器，使得脚本可以灵活地通过命令行传入参数
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--device", type=str, default="cuda:5", help="Device to use (cuda:2 or cpu)")
parser.add_argument("--vision_num_blocks", default=32, type=int, help="Number of vision blocks")
args = parser.parse_args()

vision_num_blocks = args.vision_num_blocks  # 获取vision_num_blocks参数
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 设备选择
print(device)
# 确保LoRA和线性探测不会同时启用，它们是互斥的
assert not (linear_probing and lora), "Linear probing is a subset of LoRA training procedure for ImageBind. " \
                                      "Cannot set both linear_probing=True and lora=True. "

# 根据是否使用LoRA来调整lora_factor，用于校准输出
if lora and not load_head_post_proc_finetuned:
    # 调整lora_factor以补偿训练时缺失的归一化
    lora_factor = 4 / 0.07
else:
    # 不使用LoRA时，因子为1
    lora_factor = 1

# ==================== 模型实例化与加载 ====================
# 实例化lumen6_model，并加载预训练权重
# model = lumen6_model.imagebind_huge(pretrained=True, vision_num_blocks_1=31, vision_num_blocks_2=1)
# 原始ImageBind模型实例化
model = imagebind_model.imagebind_huge(pretrained=True)

# 如果启用LoRA，则应用LoRA模块并加载其检查点
# if lora:
#     # 对模型的modality_trunks（模态主干网络）应用LoRA
#     model.modality_trunks.update(LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
#                                                                  layer_idxs=None,
#                                                                  modality_names=[ModalityType.TEXT,
#                                                                                  ModalityType.VISION]))
#     # 加载LoRA参数
#     LoRA.load_lora_modality_trunks(model.modality_trunks, checkpoint_dir=lora_dir, postfix="_trunk_last")

# 将模型设置为评估模式，并移动到指定设备
model.eval()
model.to(device)


# ==================== 推理函数定义 ====================
def run_inference():
    # 定义图像预处理流程
    data_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    # 实例化数据集和数据加载器
    test_ds = ImageNetDataset(datadir=imagenet_datadir, split="val", transform=data_transform)
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True,
                         persistent_workers=True)

    # 初始化用于存储Top-K准确率的字典
    topk1 = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 300, 400, 500, 600]
    counts_rs = {}
    for k in topk1:
        counts_rs[f'counts_r{k}'] = np.array([])

    # ==================== 批量推理循环 ====================
    with torch.no_grad():  # 在推理时禁用梯度计算，以节省内存和加速
        for batch_idx, (x, target, imgs) in enumerate(test_dl):

            x = x.to(device)
            target = [t.to(device) for t in target]

            # 准备输入数据，包括图像和文本（ImageNet的所有类别名称）
            inputs = {
                ModalityType.VISION: x,
                ModalityType.TEXT: data.load_and_transform_text(test_ds.text_list, device),
            }

            # 模型前向传播，计算多模态嵌入
            embeddings = model(inputs)

            # 计算视觉嵌入和文本嵌入之间的匹配分数
            match_value_1 = embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T * (
                lora_factor if lora else 1)

            # 计算Softmax得到概率分布
            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, -1)  # 获取最高概率的预测结果

            # 计算Top-K准确率
            top_indices_list = [torch.topk(result_1, k=k, dim=-1)[1] for k in topk1]

            # 更新Top-K正确计数
            for k, top_indices, counts_r in zip(topk1, top_indices_list, counts_rs):
                if k == 1:
                    counts_rs[counts_r] = np.concatenate([counts_rs[counts_r],
                                                          [int(predicted[i] == target[i].to(predicted.device)) for i in
                                                           range(len(predicted))]])
                else:
                    counts_rs[counts_r] = np.concatenate([counts_rs[counts_r],
                                                          [int(any(top_indices[i] == target[i].to(predicted.device)))
                                                           for i in range(len(target))]])

            # 打印当前批次的准确率信息
            logging.info(
                f"batch_idx = {batch_idx}, test_correct = {np.sum(counts_rs['counts_r1'] == 1) / len(counts_rs['counts_r1'])}, test_total = {np.sum(counts_rs['counts_r5'] == 1) / len(counts_rs['counts_r1'])}")

    # ==================== 结果汇总和保存 ====================
    indices = [i for i, value in enumerate(counts_rs['counts_r1']) if value != 0]
    print(indices)

    results = []
    lists = []
    for counts in counts_rs:
        correct = np.sum(counts_rs[counts] == 1) / len(counts_rs[counts])
        results.append(str(correct))
        lists.append(counts)

    # 打印最终的Top-K准确率结果
    print(f"Final results: {results}")


# 脚本入口点
if __name__ == "__main__":
    run_inference()