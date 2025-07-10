# train_multigpu.py
# GPU Device Setting
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # 사용할 GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score
from tqdm import tqdm
import time
import wandb
from datetime import datetime

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    get_linear_schedule_with_warmup
)

from polygon_dataset import create_polygon_datasets

# =============================================================================
# 1. 설정 및 초기화
# =============================================================================

# 기본 설정
root_dir = "/home/work/data/indo_walking/polygon_segmentation"
class_mapping_file = "/home/work/data/indo_walking/polygon_segmentation/class_mapping.txt"

# 하이퍼파라미터
EPOCHS = 100
LEARNING_RATE = 0.0005
BATCH_SIZE = 32  # 총 배치 크기 (모든 GPU에 걸쳐서)
SAVE_EVERY = 2000  # 스텝마다 모델 저장
EVAL_EVERY = 2000  # 스텝마다 평가
LOG_EVERY = 1      # 스텝마다 로그

# 모델 저장 경로
OUTPUT_DIR = "./segformer-b0-finetuned-segments-sidewalk-outputs"
MODEL_SAVE_PATH = "./trained_segformer_model"

# 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# wandb 초기화
wandb.init(
    project="segformer_segmentation",
    name=f"segformer_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "model": "nvidia/mit-b0",
        "dataset": "polygon_segmentation"
    }
)

# =============================================================================
# 2. 데이터셋 로드
# =============================================================================

print("데이터셋 로드 중...")
train_dataset, val_dataset = create_polygon_datasets(
    root_dir=root_dir,
    class_mapping_file=class_mapping_file,
    target_size=(512, 512)
)

# 클래스 정보 가져오기
id2label, label2id, num_labels = train_dataset.get_class_info()

print(f"학습 데이터셋 크기: {len(train_dataset)}")
print(f"검증 데이터셋 크기: {len(val_dataset)}")
print(f"클래스 수: {num_labels}")

# GPU 개수 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"사용 가능한 GPU 수: {num_gpus}")

# 실제 배치 크기 계산 (GPU당)
per_device_batch_size = BATCH_SIZE // num_gpus if num_gpus > 0 else BATCH_SIZE
print(f"GPU당 배치 크기: {per_device_batch_size}")

# DataLoader 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=per_device_batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=per_device_batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)

# =============================================================================
# 3. 모델 로드 및 멀티GPU 설정
# =============================================================================

print("모델 로드 중...")
pretrained_model_name = "nvidia/mit-b0"
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)

# 모델을 GPU로 이동
model = model.to(device)

# DataParallel 적용
if num_gpus > 1:
    model = nn.DataParallel(model)
    print(f"DataParallel 적용: {num_gpus}개 GPU 사용")

print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# 4. 옵티마이저 및 스케줄러 설정
# =============================================================================

# 옵티마이저 설정
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# 전체 학습 스텝 수 계산
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)  # 전체 스텝의 10%를 warmup으로 사용

# 스케줄러 설정
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"총 학습 스텝 수: {total_steps}")
print(f"Warmup 스텝 수: {warmup_steps}")

# =============================================================================
# 5. 메트릭 계산 함수
# =============================================================================

def compute_metrics(predictions, labels):
    """평가 메트릭 계산"""
    # 예측값이 로짓인 경우 argmax 적용
    if predictions.dim() == 4:  # (batch, classes, height, width)
        predictions = predictions.argmax(dim=1)
    
    # numpy로 변환
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # 정확도 계산
    accuracy = (predictions == labels).mean()
    
    # IoU 계산
    total_iou = 0
    valid_classes = 0
    
    for class_id in range(num_labels):
        pred_mask = (predictions == class_id)
        true_mask = (labels == class_id)
        
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        
        if union > 0:
            iou = intersection / union
            total_iou += iou
            valid_classes += 1
    
    mean_iou = total_iou / valid_classes if valid_classes > 0 else 0
    
    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'valid_classes': valid_classes
    }

# =============================================================================
# 6. 훈련 함수
# =============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for step, batch in enumerate(progress_bar):
        # 데이터를 GPU로 이동
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        # labels의 차원 조정 (필요한 경우)
        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        breakpoint()
        # DataParallel 사용 시 loss 평균 계산
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 손실 누적
        total_loss += loss.item()
        
        # 진행 상황 업데이트
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        # 로그 기록
        global_step = epoch * num_batches + step
        if global_step % LOG_EVERY == 0:
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch + 1,
                'step': global_step
            })
        
        # 모델 저장
        if global_step % SAVE_EVERY == 0 and global_step > 0:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss.item())
        
        # 평가 수행
        if global_step % EVAL_EVERY == 0 and global_step > 0:
            eval_metrics = evaluate_model(model, val_loader, device)
            wandb.log({
                'eval_accuracy': eval_metrics['accuracy'],
                'eval_mean_iou': eval_metrics['mean_iou'],
                'eval_valid_classes': eval_metrics['valid_classes'],
                'step': global_step
            })
            model.train()  # 평가 후 다시 학습 모드로
    
    return total_loss / num_batches

# =============================================================================
# 7. 평가 함수
# =============================================================================

def evaluate_model(model, val_loader, device):
    """모델 평가"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            if labels.dim() == 4 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # 로짓을 원본 크기로 업샘플링
            predictions = nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    # 모든 예측값과 레이블을 합침
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 메트릭 계산
    metrics = compute_metrics(all_predictions, all_labels)
    
    print(f"평가 결과:")
    print(f"  정확도: {metrics['accuracy']:.4f}")
    print(f"  평균 IoU: {metrics['mean_iou']:.4f}")
    print(f"  유효 클래스 수: {metrics['valid_classes']}")
    
    return metrics

# =============================================================================
# 8. 체크포인트 저장 함수
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss):
    """체크포인트 저장"""
    checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint-step-{step}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # 모델 상태 저장 (DataParallel 고려)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(checkpoint_path)
    
    # 추가 정보 저장
    checkpoint_info = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    
    torch.save(checkpoint_info, os.path.join(checkpoint_path, 'training_state.pth'))
    print(f"체크포인트 저장: {checkpoint_path}")

# =============================================================================
# 9. 메인 훈련 루프
# =============================================================================

def main():
    """메인 훈련 함수"""
    print("="*50)
    print("훈련 시작")
    print("="*50)
    
    best_iou = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 30)
        
        # 학습
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"평균 학습 손실: {avg_loss:.4f}")
        
        # 에포크 종료 후 평가
        eval_metrics = evaluate_model(model, val_loader, device)
        
        # wandb 로그
        wandb.log({
            'epoch_train_loss': avg_loss,
            'epoch_eval_accuracy': eval_metrics['accuracy'],
            'epoch_eval_mean_iou': eval_metrics['mean_iou'],
            'epoch': epoch + 1
        })
        
        # 최고 성능 모델 저장
        if eval_metrics['mean_iou'] > best_iou:
            best_iou = eval_metrics['mean_iou']
            print(f"새로운 최고 성능! IoU: {best_iou:.4f}")
            
            # 최고 모델 저장
            best_model_path = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(best_model_path)
    
    print("="*50)
    print("훈련 완료!")
    print(f"최고 성능 IoU: {best_iou:.4f}")
    print("="*50)
    
    # 최종 모델 저장
    final_model = model.module if hasattr(model, 'module') else model
    final_model.save_pretrained(MODEL_SAVE_PATH)
    
    # 프로세서도 저장
    processor = SegformerImageProcessor()
    processor.save_pretrained(MODEL_SAVE_PATH)
    
    print(f"최종 모델 저장 완료: {MODEL_SAVE_PATH}")
    
    # wandb 종료
    wandb.finish()

if __name__ == "__main__":
    main()