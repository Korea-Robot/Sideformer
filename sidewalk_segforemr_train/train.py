# train.py
# Multi-GPU Training을 위한 수정된 버전

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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

def setup_distributed():
    """분산 학습 환경 설정"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # CUDA 디바이스 설정
        torch.cuda.set_device(local_rank)
        
        # 분산 프로세스 그룹 초기화
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """분산 학습 환경 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    # 분산 학습 환경 설정
    rank, world_size, local_rank = setup_distributed()
    
    # 메인 프로세스에서만 출력
    is_main_process = rank == 0
    
    if is_main_process:
        print(f"분산 학습 설정:")
        print(f"- World size: {world_size}")
        print(f"- Rank: {rank}")
        print(f"- Local rank: {local_rank}")
        print(f"- 사용 가능한 GPU 수: {torch.cuda.device_count()}")
    
    # 데이터셋 설정
    root_dir = "/home/work/data/indo_walking/polygon_segmentation"
    class_mapping_file = "/home/work/data/indo_walking/polygon_segmentation/class_mapping.txt"
    
    # 데이터셋 생성
    train_dataset, val_dataset = create_polygon_datasets(
        root_dir=root_dir,
        class_mapping_file=class_mapping_file,
        target_size=(512, 512)
    )
    
    if is_main_process:
        sample = train_dataset[0]
        print(f"이미지 경로: {sample['image_path']}")
        print(f"마스크 경로: {sample['mask_path']}")
        print(f"이미지 크기: {sample['pixel_values'].shape}")
        print(f"마스크 크기: {sample['labels'].shape}")
    
    # 클래스 정보 가져오기
    id2label, label2id, num_labels = train_dataset.get_class_info()
    
    if is_main_process:
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
    
    if is_main_process:
        print(f"모델 로드 완료: {pretrained_model_name}")
        print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 평가 메트릭 함수
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
    
    # 훈련 설정
    epochs = 100
    lr = 0.0005
    # Multi-GPU를 위한 배치 크기 조정 (전체 배치 크기를 GPU 수로 나눔)
    per_device_batch_size = 16
    
    if is_main_process:
        print(f"\n훈련 설정:")
        print(f"- 에포크 수: {epochs}")
        print(f"- 학습률: {lr}")
        print(f"- GPU당 배치 크기: {per_device_batch_size}")
        print(f"- 전체 배치 크기: {per_device_batch_size * world_size}")
    
    # TrainingArguments 설정 (Multi-GPU 지원)
    training_args = TrainingArguments(
        output_dir="./segformer-b0-finetuned-segments-sidewalk-outputs",
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        save_total_limit=3,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        logging_steps=10,
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="wandb" if is_main_process else None,  # 메인 프로세스에서만 로깅
        run_name="segformer_segmentation_experiment-0703-multi-gpu",
        logging_dir="./logs",
        dataloader_pin_memory=False,
        # Multi-GPU 관련 설정
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,  # 데이터 로더 워커 수
        # 그래디언트 체크포인팅으로 메모리 사용량 줄이기
        gradient_checkpointing=True,
        # 메모리 효율성을 위한 설정
        fp16=True,  # Mixed precision training
        ddp_backend="nccl",
        # 로깅 관련 설정
        logging_first_step=True,
        save_on_each_node=False,
        # 평가 관련 설정
        eval_accumulation_steps=5,
    )
    
    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    if is_main_process:
        print("Multi-GPU 훈련을 시작합니다...")
        print("=" * 50)
    
    # 훈련 실행
    training_result = trainer.train()
    
    if is_main_process:
        print("=" * 50)
        print("훈련 완료!")
        print(f"최종 훈련 손실: {training_result.training_loss:.4f}")
        
        # 모델 저장 (메인 프로세스에서만)
        model.save_pretrained("./trained_segformer_model")
        processor.save_pretrained("./trained_segformer_model")
        print("모델이 './trained_segformer_model' 디렉토리에 저장되었습니다.")
    
    # 분산 학습 환경 정리
    cleanup_distributed()

if __name__ == "__main__":
    main()