# GPU Device Setting

# pytorch나 transformer 라이브러리 보다 먼저 이것을 가으와져오야함.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# https://discuss.huggingface.co/t/setting-specific-device-for-trainer/784/26
# SegFormer 모델 학습을 위한 완전한 코드
import json
import torch
# torch.cuda.set_device(1)


from torch import nn
import evaluate
from datasets import load_dataset
from torchvision.transforms import ColorJitter
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer
)
from huggingface_hub import hf_hub_download


from huggingface_trainer.polygon_dataset import create_polygon_datasets

# 테스트 코드
root_dir = "/home/work/data/indo_walking/polygon_segmentation"
class_mapping_file = "/home/work/data/indo_walking/polygon_segmentation/class_mapping.txt"

# 데이터셋 생성
train_dataset, val_dataset = create_polygon_datasets(
    root_dir=root_dir,
    class_mapping_file=class_mapping_file,
    target_size=(512, 512)
)

# 첫 번째 샘플 확인
sample = train_dataset[0]
print(f"이미지 경로: {sample['image_path']}")
print(f"마스크 경로: {sample['mask_path']}")
print(f"이미지 크기: {sample['pixel_values'].shape}")


print(f"마스크 크기: {sample['labels'].shape}")

breakpoint()
# 클래스 정보 출력
id2label, label2id, num_labels = train_dataset.get_class_info()
print(f"\n클래스 수: {num_labels}")
print("클래스 매핑:")
for class_id, class_name in list(id2label.items())[:10]:  # 처음 10개만 출력
    print(f"  {class_id}: {class_name}")

breakpoint()

"""
(Pdb) sample['pixel_values'].shape
torch.Size([3, 512, 512])
(Pdb) sample['labels'].shape
torch.Size([1, 512, 512])
"""

# # 샘플 시각화 (matplotlib 필요)
# try:
#     train_dataset.visualize_sample(0, "sample_visualization.png")
#     print("\n샘플 이미지가 'sample_visualization.png'로 저장되었습니다.")
# except ImportError:
#     print("\n시각화를 위해서는 matplotlib가 필요합니다.")

# breakpoint()