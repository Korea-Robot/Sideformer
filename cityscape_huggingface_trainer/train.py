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



# =============================================================================
# 1. 데이터셋 설정 및 레이블 정보 로드
# =============================================================================

hf_dataset_identifier = "segments/sidewalk-semantic"

# 데이터셋 로드 
ds = load_dataset(hf_dataset_identifier)
ds = ds.shuffle(seed=1)
ds = ds["train"].train_test_split(test_size=0.1)


################# 

train_ds = ds["train"]
test_ds = ds["test"]


print((train_ds.shape))             # (900, 2)





print(f"훈련 데이터: {len(train_ds)}개")
print(f"테스트 데이터: {len(test_ds)}개")


"""
훈련 데이터: 800개
테스트 데이터: 200개
"""

# 레이블 정보 로드
repo_id = f"datasets/{hf_dataset_identifier}"
filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

print(f"클래스 수: {num_labels}")
print(f"클래스 레이블: {id2label}")

"""
클래스 수: 35
클래스 레이블: {0: 'unlabeled', 1: 'flat-road', 2: 'flat-sidewalk', 3: 'flat-crosswalk', 4: 'flat-cyclinglane', 5: 'flat-parkingdriveway', 6: 'flat-railtrack', 7: 'flat-curb', 8: 'human-person', 9: 'human-rider', 10: 'vehicle-car', 11: 'vehicle-truck', 12: 'vehicle-bus', 13: 'vehicle-tramtrain', 14: 'vehicle-motorcycle', 15: 'vehicle-bicycle', 16: 'vehicle-caravan', 17: 'vehicle-cartrailer', 18: 'construction-building', 19: 'construction-door', 20: 'construction-wall', 21: 'construction-fenceguardrail', 22: 'construction-bridge', 23: 'construction-tunnel', 24: 'construction-stairs', 25: 'object-pole', 26: 'object-trafficsign', 27: 'object-trafficlight', 28: 'nature-vegetation', 29: 'nature-terrain', 30: 'sky', 31: 'void-ground', 32: 'void-dynamic', 33: 'void-static', 34: 'void-unclear'}
데이터 변환 설정 완료
Some weights of SegformerForSemanticSegmentation were not initialized from the model checkpoint at nvidia/mit-b0 and are newly initialized: ['decode_head.batch_norm.bias', 'decode_head.batch_norm.num_batches_tracked', 'decode_head.batch_norm.running_mean', 'decode_head.batch_norm.running_var', 'decode_head.batch_norm.weight', 'decode_head.classifier.bias', 'decode_head.classifier.weight', 'decode_head.linear_c.0.proj.bias', 'decode_head.linear_c.0.proj.weight', 'decode_head.linear_c.1.proj.bias', 'decode_head.linear_c.1.proj.weight', 'decode_head.linear_c.2.proj.bias', 'decode_head.linear_c.2.proj.weight', 'decode_head.linear_c.3.proj.bias', 'decode_head.linear_c.3.proj.weight', 'decode_head.linear_fuse.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
모델 로드 완료: nvidia/mit-b0
모델 파라미터 수: 3,723,139

"""

# =============================================================================
# 2. 이미지 프로세서 및 데이터 변환 설정
# =============================================================================

processor = SegformerImageProcessor()

# data augmentation!!
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

def train_transforms(example_batch):
    """훈련용 데이터 변환 (데이터 증강 포함)"""
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

def val_transforms(example_batch):
    """검증용 데이터 변환"""
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

# 데이터셋에 변환 함수 적용  : augmentation적용용
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

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

print("모델이 './trained_segformer_model' 디렉토리에 저장되었습니다.")

# Hub에 푸시하려면 아래 주석을 해제
# kwargs = {
#     "tags": ["vision", "image-segmentation"],
#     "finetuned_from": pretrained_model_name,
#     "dataset": hf_dataset_identifier,
# }
# processor.push_to_hub(hub_model_id)
# trainer.push_to_hub(**kwargs)

