# train_accelerate.py
# Accelerate를 사용한 간단한 Multi-GPU 학습

import os
import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score
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
from polygon_dataset import create_polygon_datasets

# GPU 설정 - 원래 코드 유지
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

def main():
    # 데이터셋 설정
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
    
    # 클래스 정보 출력
    id2label, label2id, num_labels = train_dataset.get_class_info()
    print(f"\n클래스 수: {num_labels}")
    print("클래스 매핑:")
    for class_id, class_name in list(id2label.items())[:10]:
        print(f"  {class_id}: {class_name}")
    
    # 모델 로드
    processor = SegformerImageProcessor()
    pretrained_model_name = "nvidia/mit-b0"
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id
    )
    
    print(f"모델 로드 완료: {pretrained_model_name}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
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
            
            # labels의 차원이 4D인 경우 (batch_size, 1, height, width) -> (batch_size, height, width)로 변환
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            
            # 직접 IoU 계산
            num_labels = len(id2label)
            total_iou = 0
            valid_classes = 0
            
            for class_id in range(num_labels):
                # 각 클래스별로 IoU 계산
                pred_mask = (pred_labels == class_id)
                true_mask = (labels == class_id)
                
                intersection = np.logical_and(pred_mask, true_mask).sum()
                union = np.logical_or(pred_mask, true_mask).sum()
                
                if union > 0:  # 해당 클래스가 존재하는 경우만 계산
                    iou = intersection / union
                    total_iou += iou
                    valid_classes += 1
            
            # 전체 정확도 계산
            accuracy = (pred_labels == labels).mean()
            
            # 평균 IoU 계산
            mean_iou = total_iou / valid_classes if valid_classes > 0 else 0
            
            return {
                'mean_iou': mean_iou,
                'accuracy': accuracy,
                'valid_classes': valid_classes
            }
    
    # 하이퍼파라미터 설정
    epochs = 100
    lr = 0.0005
    batch_size = 32
    
    print(f"\n훈련 설정:")
    print(f"- 에포크 수: {epochs}")
    print(f"- 학습률: {lr}")
    print(f"- GPU당 배치 크기: {batch_size}")
    print(f"- 전체 배치 크기: {batch_size * torch.cuda.device_count()}")
    
    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir="./segformer-b0-finetuned-segments-sidewalk-outputs",
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=3,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        logging_steps=10,
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="wandb",
        run_name="segformer_segmentation_experiment-0703-accelerate",
        logging_dir="./logs",
        dataloader_pin_memory=False,
        dataloader_num_workers=6,
        gradient_checkpointing=False,
        fp16=True,  # Mixed precision training
        eval_accumulation_steps=5,
        # Accelerate에서 자동으로 처리되므로 DDP 관련 설정 불필요
    )
    
    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Multi-GPU 훈련을 시작합니다...")
    print("=" * 50)
    
    # 훈련 실행
    training_result = trainer.train()
    
    print("=" * 50)
    print("훈련 완료!")
    print(f"최종 훈련 손실: {training_result.training_loss:.4f}")
    
    # 모델 저장
    model.save_pretrained("./trained_segformer_model")
    processor.save_pretrained("./trained_segformer_model")
    print("모델이 './trained_segformer_model' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()