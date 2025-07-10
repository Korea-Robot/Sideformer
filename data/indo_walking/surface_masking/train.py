# GPU Device Setting

# pytorch나 transformer 라이브러리 보다 먼저 이것을 가으와져오야함.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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

from surface_dataset import SurfaceSegmentationDataset
# --- 설정 ---
# 전처리된 데이터 정보가 담긴 metadata.json 파일 경로
metadata_path = "./processed_dataset/metadata.json"

# 학습 파라미터
batch_size = 32
target_size = (512, 512)
num_workers = 16  # CPU 코어 수에 맞게 조절

"""
metadata.json을 읽어 학습/검증 데이터로더를 생성
"""
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 데이터셋 인스턴스 생성
train_ds = SurfaceSegmentationDataset(metadata['train_data'], target_size, is_train=True)
test_ds = SurfaceSegmentationDataset(metadata['valid_data'], target_size, is_train=False)


filename = "id2label.json"

# 클래스 이름과 인덱스 매핑 (배경은 0)
class_to_idx = {
    'background': 0,
    'caution_zone': 1,
    'bike_lane': 2,
    'alley': 3,
    'roadway': 4,
    'braille_guide_blocks': 5,
    'sidewalk': 6
}

# id2label = ????
# id2label = {int(k): v for k, v in id2label.items()}
# label2id = {v: k for k, v in id2label.items()}
# num_labels = len(id2label)

# class_to_idx로부터 id2label과 label2id를 생성합니다.
id2label = {int(idx): label for label, idx in class_to_idx.items()}
label2id = class_to_idx
num_labels = len(id2label)

print(f"클래스 수: {num_labels}")
print(f"클래스 레이블: {id2label}")


# =============================================================================
# 2. 이미지 프로세서 및 데이터 변환 설정
# =============================================================================

processor = SegformerImageProcessor()

# data augmentation!!
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

print("데이터 변환 설정 완료")

# =============================================================================
# 3. 모델 로드
# =============================================================================

pretrained_model_name = "nvidia/mit-b0"
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)

print(f"모델 로드 완료: {pretrained_model_name}")
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# 4. 평가 메트릭 설정
# =============================================================================

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    """평가 메트릭 계산 함수"""
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        
        # 로짓을 레이블 크기로 업스케일링
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=processor.do_reduce_labels,
        )
        
        # 카테고리별 메트릭 추가
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
        
        return metrics

# =============================================================================
# 5. 훈련 설정
# =============================================================================

# 하이퍼파라미터 설정
epochs = 100
lr = 0.00006
batch_size = 64

print()
print('batch size! ',batch_size)
print()

# 모델 저장 경로 및 허브 ID 설정 (필요시 수정)
hub_model_id = "segformer-b0-finetuned-segments-sidewalk-custom"

training_args = TrainingArguments(
    output_dir="./segformer-b0-finetuned-segments-sidewalk-outputs",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    eval_strategy="steps",  # evaluation_strategy -> eval_strategy
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True,
    push_to_hub=False,  # Hub에 푸시하려면 True로 변경
    hub_model_id=hub_model_id,
    hub_strategy="end",
    report_to="wandb",  # wandb 등 로깅 도구 사용하지 않음
    run_name="segformer_segmentation_experiment",
    logging_dir="./logs",
    dataloader_pin_memory=False,  # 메모리 사용량 최적화
)

print(f"훈련 설정:")
print(f"- 에포크 수: {epochs}")
print(f"- 학습률: {lr}")
print(f"- 배치 크기: {batch_size}")
print(f"- 출력 디렉토리: {training_args.output_dir}")

# =============================================================================
# 6. Trainer 설정 및 훈련 시작
# =============================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

print("훈련을 시작합니다...")
print("=" * 50)

# 훈련 실행
training_result = trainer.train()

print("=" * 50)
print("훈련 완료!")
print(f"최종 훈련 손실: {training_result.training_loss:.4f}")

# =============================================================================
# 7. 모델 저장
# =============================================================================

# 로컬에 모델 저장
model.save_pretrained("./trained_segformer_model")
processor.save_pretrained("./trained_segformer_model")


