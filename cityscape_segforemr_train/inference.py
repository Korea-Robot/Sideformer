import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import torch
from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
from datasets import load_dataset
import json 

from huggingface_hub import hf_hub_download



hf_dataset_identifier = "segments/sidewalk-semantic"

# 데이터셋 로드 
ds = load_dataset(hf_dataset_identifier)
ds = ds.shuffle(seed=1)
ds = ds["train"].train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]

# 레이블 정보 로드
repo_id = f"datasets/{hf_dataset_identifier}"
filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)


processor = SegformerImageProcessor()


def val_transforms(example_batch):
    """검증용 데이터 변환"""
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

train_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

print("데이터 변환 설정 완료")

# pretrained_model_name = "nvidia/mit-b0"
# model = SegformerForSemanticSegmentation.from_pretrained(
#     pretrained_model_name,
#     id2label=id2label,
#     label2id=label2id
# )

model_dir = "./trained_segformer_model2"

# 모델 로드
model = SegformerForSemanticSegmentation.from_pretrained(model_dir)

# Processor 로드
processor = SegformerImageProcessor.from_pretrained(model_dir)



print("\n추론 예제를 실행합니다...")

# breakpoint()

# 테스트 이미지 로드
for i in range(10):
    # test_image = torch.tensor(test_ds[i]['pixel_values']).unsqueeze(0)
    test_image = torch.tensor(train_ds[i]['pixel_values']).unsqueeze(0)
    # gt_seg = torch.tensor(test_ds[i]['labels']).unsqueeze(0)
    gt_seg = torch.tensor(train_ds[i]['labels']).unsqueeze(0)

    # 추론 수행
    # inputs = processor(images=test_image, return_tensors="pt")

    inputs = {
        "pixel_values": test_image,   # shape: (1, 3, 512, 512)
        "labels": gt_seg              # shape: (1, 512, 512)
    }


    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    import torch.nn as nn


    # 로짓을 원본 이미지 크기로 업샘플링
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=test_image.shape[-2:],  # (height, width)
        mode='bilinear',
        align_corners=False
    )

    # 예측 결과 생성
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    print(f"추론 완료!")
    print(f"예측 세그멘테이션 크기: {pred_seg.shape}")
    print(f"고유한 예측 클래스: {torch.unique(pred_seg).tolist()}")

    # =============================================================================
    # 9. 결과 시각화 (옵션)
    # =============================================================================

    # 시각화를 원하면 아래 코드를 사용하세요
    # """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 원본 이미지
    axes[0].imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 실제 세그멘테이션
    axes[1].imshow(gt_seg.squeeze().cpu().numpy(), cmap='tab20')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # 예측 세그멘테이션
    axes[2].imshow(pred_seg.cpu().numpy(), cmap='tab20')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'outputs/train_segmentation_result2_{i}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    # """

    print("\n훈련 및 추론이 모두 완료되었습니다!")
    print("시각화를 원하시면 코드 하단의 주석을 해제하고 matplotlib을 설치하세요.")
