# hyperparameter_optimization.py - 하이퍼파라미터 최적화를 위한 DDP 학습 코드

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import logging
from datetime import datetime, timedelta
import optuna
from optuna.integration import WeightsAndBiasesCallback
import pickle
from pathlib import Path
from sklearn.metrics import f1_score, jaccard_score
import time
from typing import Dict, List, Tuple, Optional

# 이전 단계에서 작성한 PolygonSegmentationDataset 클래스를 임포트합니다.
# 이 파일이 polygon_dataset.py와 같은 디렉토리에 있다고 가정합니다.
from polygon_dataset import PolygonSegmentationDataset

class AdvancedMetrics:
    """고급 평가 메트릭 계산"""
    
    def __init__(self, num_classes: int, ignore_index: int = -100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """메트릭 초기화"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_pixels = 0
        
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """메트릭 업데이트"""
        # GPU 텐서를 CPU로 이동
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        # ignore_index 제외
        mask = target_np != self.ignore_index
        pred_np = pred_np[mask]
        target_np = target_np[mask]
        
        # Confusion matrix 업데이트
        # np.add.at을 사용하여 더 효율적으로 업데이트
        np.add.at(self.confusion_matrix, (target_np, pred_np), 1)
        
        self.total_pixels += len(pred_np)
    
    def compute_metrics(self) -> Dict[str, float]:
        """메트릭 계산"""
        cm = self.confusion_matrix
        
        # 분산 환경에서 모든 프로세스의 confusion matrix를 합산
        if dist.is_initialized():
            cm_tensor = torch.tensor(cm, dtype=torch.float64, device=f'cuda:{dist.get_rank()}')
            dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)
            cm = cm_tensor.cpu().numpy()

        # Pixel Accuracy
        pixel_acc = np.diag(cm).sum() / (cm.sum() + 1e-10)
        
        # Mean Accuracy (각 클래스별 정확도의 평균)
        class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
        mean_acc = np.nanmean(class_acc)
        
        # IoU 계산
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = intersection / (union + 1e-10)
        mean_iou = np.nanmean(iou)
        
        # F1 Score 계산
        precision = intersection / (cm.sum(axis=0) + 1e-10)
        recall = intersection / (cm.sum(axis=1) + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        mean_f1 = np.nanmean(f1)
        
        return {
            'pixel_accuracy': pixel_acc,
            'mean_accuracy': mean_acc,
            'mean_iou': mean_iou,
            'mean_f1': mean_f1,
            'per_class_iou': iou,
            'per_class_f1': f1,
            'per_class_accuracy': class_acc
        }

def setup_logging(rank: int, log_dir: str = "logs"):
    """로깅 설정"""
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - RANK:%(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
                logging.StreamHandler()
            ]
        )
        logging.getLogger().name = str(rank)
    else:
        # 다른 프로세스에서는 경고 이상의 로그만 출력하도록 설정
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - RANK:%(name)s - %(levelname)s - %(message)s')
        logging.getLogger().name = str(rank)



def setup_ddp(rank: int, world_size: int, master_addr: str = 'localhost', master_port: str = '12355'):
    """DDP 환경 설정"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    
    # 시스템 및 네트워크 환경에 따라 주석 처리 또는 수정
    # os.environ['NCCL_P2P_DISABLE'] = '1'
    # os.environ['NCCL_IB_DISABLE'] = '1'
    # os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # 로컬 인터페이스 사용

    torch.cuda.set_device(rank)
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=3600)  # 1시간 타임아웃
    )
    
    return rank

def cleanup():
    """DDP 환경 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_model_with_config(model_name: str, num_labels: int, dropout: float = 0.1) -> SegformerForSemanticSegmentation:
    """설정 가능한 모델 생성"""
    config = SegformerConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    config.classifier_dropout_prob = dropout
    
    # 사전 훈련된 가중치 로드 (분류기 헤드 제외)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True # num_labels가 다른 head는 무시
    )
    
    return model

def get_optimizer_and_scheduler(model, config: Dict, total_steps: int):
    """옵티마이저와 스케줄러 생성"""
    # 파라미터 그룹 분리 (backbone과 head 다른 학습률)
    backbone_params = list(model.module.segformer.parameters())
    head_params = list(model.module.decode_head.parameters())
    
    param_groups = [
        {'params': backbone_params, 'lr': config['backbone_lr']},
        {'params': head_params, 'lr': config['head_lr']}
    ]
    
    # 옵티마이저 선택
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=config.get('momentum', 0.9), # get으로 안전하게 접근
            weight_decay=config['weight_decay'],
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # 스케줄러 선택
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=config['backbone_lr'] * 0.01
        )
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=total_steps // 3,
            gamma=0.1
        )
    elif config['scheduler'] == 'poly':
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=total_steps,
            power=0.9
        )
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")
    
    return optimizer, scheduler

def validate_model_advanced(model, valid_loader, device, num_classes):
    """향상된 검증 함수"""
    model.eval()
    metrics = AdvancedMetrics(num_classes)
    val_loss = 0.0
    
    with torch.no_grad():
        for data in valid_loader:
            images = data['pixel_values'].to(device, non_blocking=True)
            masks = data['labels'].to(device, non_blocking=True)
            
            inputs = {'pixel_values': images, 'labels': masks}
            outputs = model(**inputs)
            
            val_loss += outputs.loss.item()
            
            # 예측 결과 계산
            logits = outputs.logits
            upsampled = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            predictions = torch.argmax(upsampled, dim=1)
            metrics.update(predictions, masks)
    
    # 손실 값 동기화
    val_loss_tensor = torch.tensor(val_loss, device=device)
    if dist.is_initialized():
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    
    avg_val_loss = val_loss_tensor.item() / (len(valid_loader.dataset))
    
    # 메트릭 계산 (내부에서 all_reduce 수행)
    computed_metrics = metrics.compute_metrics()
    computed_metrics['val_loss'] = avg_val_loss
    
    return computed_metrics

def objective(trial, args, rank, world_size):
    """Optuna 최적화 목적 함수 (단일 프로세스 내에서 실행)"""
    # 하이퍼파라미터 샘플링 (모든 프로세스에서 동일하게 수행)
    config = {
        'backbone_lr': trial.suggest_float('backbone_lr', 1e-6, 1e-4, log=True),
        'head_lr': trial.suggest_float('head_lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16]),
        'optimizer': trial.suggest_categorical('optimizer', ['adamw']), # 'sgd'는 성능이 낮아 제외
        'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'poly']),
        'epochs': args.optuna_epochs,
    }
    # SGD일 경우에만 momentum 샘플링
    if config['optimizer'] == 'sgd':
       config['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)

    # 모델 학습
    best_metric = train_single_process(rank, world_size, config, args, trial)
    
    return best_metric

def train_single_process(rank, world_size, config, args, trial=None):
    """단일 프로세스에서 실행되는 학습 함수"""
    try:
        # DDP 설정 및 로깅
        setup_ddp(rank, world_size, args.master_addr, args.master_port)
        device = torch.device(f'cuda:{rank}')
        setup_logging(rank, args.log_dir)
        
        if rank == 0:
            logging.info(f"Starting Trial {trial.number if trial else 'Final'} with config: {config}")
        
        # 데이터 로드
        class_to_idx = {
            'background': 0, 'barricade': 1, 'bench': 2, 'bicycle': 3, 'bollard': 4,
            'bus': 5, 'car': 6, 'carrier': 7, 'cat': 8, 'chair': 9, 'dog': 10,
            'fire_hydrant': 11, 'kiosk': 12, 'motorcycle': 13, 'movable_signage': 14,
            'parking_meter': 15, 'person': 16, 'pole': 17, 'potted_plant': 18,
            'power_controller': 19, 'scooter': 20, 'stop': 21, 'stroller': 22,
            'table': 23, 'traffic_light': 24, 'traffic_light_controller': 25,
            'traffic_sign': 26, 'tree_trunk': 27, 'truck': 28, 'wheelchair': 29
        }
        num_labels = len(class_to_idx)
        
        # 데이터셋 생성
        train_dataset = PolygonSegmentationDataset(root_dir=args.data_dir, is_train=True)
        valid_dataset = PolygonSegmentationDataset(root_dir=args.data_dir, is_train=False)
        
        # DDP 샘플러
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        # 데이터로더
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            sampler=train_sampler,
            num_workers=args.num_workers, 
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=config['batch_size'], 
            sampler=valid_sampler,
            num_workers=args.num_workers, 
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
        
        # 모델 생성 및 DDP 래핑
        model = create_model_with_config(args.model_name, num_labels, config['dropout']).to(device)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        
        # 옵티마이저 및 스케줄러
        total_steps = len(train_loader) * config['epochs']
        optimizer, scheduler = get_optimizer_and_scheduler(model, config, total_steps)
        
        # 학습 루프
        best_metric = float('inf') # val_loss를 최소화하는 것을 목표로 함
        patience = 10
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            train_sampler.set_epoch(epoch)
            model.train()
            train_loss = 0.0
            
            for i, data in enumerate(train_loader):
                images = data['pixel_values'].to(device, non_blocking=True)
                masks = data['labels'].to(device, non_blocking=True)
                
                inputs = {'pixel_values': images, 'labels': masks}
                outputs = model(**inputs)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step() # Step-based scheduler
                
                train_loss += loss.item()

            # 검증 (모든 프로세스에서 수행, 결과는 rank 0에서 집계)
            metrics = validate_model_advanced(model, valid_loader, device, num_labels)
            
            if rank == 0:
                current_metric = metrics['val_loss']
                
                logging.info(f"Epoch {epoch}: Val Loss={metrics['val_loss']:.4f}, "
                             f"mIoU={metrics['mean_iou']:.4f}, "
                             f"Pixel Acc={metrics['pixel_accuracy']:.4f}")

                # Optuna pruning
                if trial:
                    trial.report(current_metric, epoch)
                    if trial.should_prune():
                        cleanup()
                        raise optuna.exceptions.TrialPruned()

                # Early stopping
                if current_metric < best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break
        
        # DDP 환경 정리
        cleanup()
        return best_metric
        
    except Exception as e:
        if rank == 0:
            logging.error(f"Training error: {e}")
        cleanup()
        # Optuna에서 예외 발생 시, 해당 trial 실패 처리
        if isinstance(e, optuna.exceptions.TrialPruned):
             raise
        return float('inf')


def objective_worker(rank, world_size, args, study, n_trials_per_process):
    """각 프로세스에서 Optuna objective를 실행하는 워커"""
    for _ in range(n_trials_per_process):
        # rank 0에서만 trial을 생성하고, 다른 프로세스와 동기화
        trial_params = None
        if rank == 0:
            trial = study.ask()
            trial_params = trial.params
        
        # trial 파라미터를 다른 프로세스로 브로드캐스팅
        params_list = [trial_params]
        dist.broadcast_object_list(params_list, src=0)
        
        if rank != 0:
            # 다른 프로세스는 받은 파라미터로 trial을 재생성
            trial = optuna.trial.create_trial(
                params=params_list[0],
                distributions=study.sampler.infer_relative_search_space(study, None), # DDP에서는 FixedTrial 사용이 더 복잡
                value=None
            )

        try:
            value = objective(trial, args, rank, world_size)
            if rank == 0:
                study.tell(trial, value)
        except optuna.exceptions.TrialPruned:
            if rank == 0:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception as e:
            if rank == 0:
                logging.error(f"Trial failed with error: {e}")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)


def run_hyperparameter_optimization(args):
    """하이퍼파라미터 최적화 실행 (DDP 환경에서)"""
    # Optuna 스터디 생성 (rank 0에서만)
    study = None
    if args.rank == 0:
        study_name = f"segmentation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(args.study_dir, exist_ok=True)
        storage_path = os.path.join(args.study_dir, f"{study_name}.db")
        
        study = optuna.create_study(
            direction='minimize',  # val_loss 최소화
            study_name=study_name,
            storage=f'sqlite:///{storage_path}',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
        
        logging.info(f"Starting hyperparameter optimization with {args.n_trials} trials...")

    # 각 프로세스에 할당될 trial 수
    # n_trials_per_process = args.n_trials // args.world_size
    # if args.rank < args.n_trials % args.world_size:
    #     n_trials_per_process += 1
    
    # objective_worker(args.rank, args.world_size, args, study, n_trials_per_process)

    # Optuna DDP 통합은 복잡하므로, 단일 GPU로 최적화를 진행하는 것이 더 안정적일 수 있음
    # 아래는 단일 GPU (rank 0)에서만 최적화를 수행하는 방식
    if args.rank == 0:
        study.optimize(lambda trial: objective(trial, args, args.rank, 1), # world_size=1로 단일 실행
                       n_trials=args.n_trials)
        
        # 결과 출력
        logging.info("\nOptimization completed!")
        logging.info(f"Best trial: {study.best_trial.number}")
        logging.info(f"Best value (val_loss): {study.best_value:.4f}")
        logging.info(f"Best params: {study.best_params}")
        
        # 최적 파라미터 저장
        best_params_path = os.path.join(args.study_dir, "best_params.json")
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        
        return study.best_params
    else:
        # 다른 rank들은 대기
        return None


def train_final_model_worker(rank, world_size, config, args):
    """최종 모델 학습을 위한 워커"""
    try:
        # DDP 설정 및 로깅
        setup_ddp(rank, world_size, args.master_addr, args.master_port)
        device = torch.device(f'cuda:{rank}')
        setup_logging(rank, args.log_dir)

        if rank == 0:
            logging.info(f"Final training with config: {config}")
        
        # 데이터 로드
        class_to_idx = {
            'background': 0, 'barricade': 1, 'bench': 2, 'bicycle': 3, 'bollard': 4,
            'bus': 5, 'car': 6, 'carrier': 7, 'cat': 8, 'chair': 9, 'dog': 10,
            'fire_hydrant': 11, 'kiosk': 12, 'motorcycle': 13, 'movable_signage': 14,
            'parking_meter': 15, 'person': 16, 'pole': 17, 'potted_plant': 18,
            'power_controller': 19, 'scooter': 20, 'stop': 21, 'stroller': 22,
            'table': 23, 'traffic_light': 24, 'traffic_light_controller': 25,
            'traffic_sign': 26, 'tree_trunk': 27, 'truck': 28, 'wheelchair': 29
        }
        num_labels = len(class_to_idx)
        
        # 전체 데이터셋 (train + valid) 사용
        train_dataset = PolygonSegmentationDataset(root_dir=args.data_dir, is_train=True)
        valid_dataset = PolygonSegmentationDataset(root_dir=args.data_dir, is_train=False)
        full_dataset = ConcatDataset([train_dataset, valid_dataset])
        
        # DDP 샘플러
        full_sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
        # 데이터로더
        full_loader = DataLoader(
            full_dataset, 
            batch_size=config['batch_size'], 
            sampler=full_sampler,
            num_workers=args.num_workers, 
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
        
        # 모델 생성 및 DDP 래핑
        model = create_model_with_config(args.model_name, num_labels, config['dropout']).to(device)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        
        # 옵티마이저 및 스케줄러
        total_steps = len(full_loader) * config['epochs']
        optimizer, scheduler = get_optimizer_and_scheduler(model, config, total_steps)
        
        # WandB 초기화 (rank 0에서만)
        if rank == 0 and args.use_wandb:
            wandb.init(
                project=f"{args.wandb_project}_final",
                name=f"final_model_{datetime.now().strftime('%m%d_%H%M')}",
                config=config
            )
        
        # 학습 루프
        for epoch in range(config['epochs']):
            full_sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            
            for i, data in enumerate(full_loader):
                images = data['pixel_values'].to(device, non_blocking=True)
                masks = data['labels'].to(device, non_blocking=True)
                
                inputs = {'pixel_values': images, 'labels': masks}
                outputs = model(**inputs)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if rank == 0 and i % args.log_interval == 0:
                    logging.info(f"Epoch [{epoch+1}/{config['epochs']}], "
                                 f"Step [{i+1}/{len(full_loader)}], "
                                 f"Loss: {loss.item():.4f}")
            
            # Epoch 정보 로깅
            if rank == 0:
                avg_loss = epoch_loss / len(full_loader)
                logging.info(f"Epoch [{epoch+1}/{config['epochs']}] - Avg Loss: {avg_loss:.4f}")
                
                if args.use_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "learning_rate_backbone": scheduler.get_last_lr()[0],
                        "learning_rate_head": scheduler.get_last_lr()[1],
                    })
                
                # 모델 저장
                if (epoch + 1) % args.save_interval == 0:
                    os.makedirs(args.final_model_dir, exist_ok=True)
                    save_path = os.path.join(args.final_model_dir, f"final_model_epoch_{epoch+1}.pth")
                    torch.save(model.module.state_dict(), save_path)
                    logging.info(f"Model saved to {save_path}")

        # 최종 모델 저장
        if rank == 0:
            os.makedirs(args.final_model_dir, exist_ok=True)
            final_path = os.path.join(args.final_model_dir, "final_model.pth")
            torch.save(model.module.state_dict(), final_path)
            
            # 설정 저장
            with open(os.path.join(args.final_model_dir, "final_config.json"), 'w') as f:
                json.dump(config, f, indent=4)
            
            logging.info(f"Final model saved successfully to {final_path}!")
            
            if args.use_wandb:
                wandb.finish()
        
        cleanup()
        
    except Exception as e:
        if rank == 0:
            logging.error(f"Final training error: {e}")
        cleanup()


def train_final_model(best_params: Dict, args):
    """최적 파라미터로 전체 데이터에서 최종 모델 학습을 시작하는 함수"""
    logging.info("Training final model with best parameters...")
    
    # 최적 파라미터에 전체 에폭 수 설정
    final_config = best_params.copy()
    final_config['epochs'] = args.final_epochs
    final_config['batch_size'] = best_params.get('batch_size', 8) # 배치사이즈 고정 또는 best_params에서 가져오기

    # 멀티프로세싱으로 최종 학습 실행
    import torch.multiprocessing as mp
    mp.spawn(train_final_model_worker, 
             args=(args.world_size, final_config, args), 
             nprocs=args.world_size, 
             join=True)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for Segmentation')
    
    # 데이터 관련
    parser.add_argument('--data_dir', type=str, 
                        default="/home/work/data/indo_walking/polygon_segmentation",
                        help='데이터 디렉토리 경로')
    
    # 모델 관련
    parser.add_argument('--model_name', type=str, 
                        default="nvidia/mit-b0",
                        help='사용할 Segformer 모델 이름 (e.g., nvidia/mit-b0, nvidia/mit-b5)')
    
    # 하이퍼파라미터 최적화 관련
    parser.add_argument('--run_hpo', action='store_true', help='하이퍼파라미터 최적화를 실행합니다.')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Optuna 시도 횟수')
    parser.add_argument('--optuna_epochs', type=int, default=20,
                        help='HPO의 각 시도별 학습 에폭 수')
    parser.add_argument('--study_dir', type=str, default="studies",
                        help='Optuna 스터디 저장 디렉토리')
    parser.add_argument('--best_params_path', type=str, default="studies/best_params.json",
                        help='최종 학습에 사용할 최적 하이퍼파라미터 파일 경로')

    # 최종 학습 관련
    parser.add_argument('--run_final_training', action='store_true', help='최적 파라미터로 최종 모델을 학습합니다.')
    parser.add_argument('--final_epochs', type=int, default=100,
                        help='최종 모델 학습 에폭 수')
    
    # 학습 환경 관련
    parser.add_argument('--num_workers', type=int, default=4,
                        help='데이터 로더 워커 수')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='학습 중 로그 출력 간격 (스텝 단위)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='최종 학습 중 모델 저장 간격 (에폭 단위)')
    
    # 저장 경로 관련
    parser.add_argument('--log_dir', type=str, default="logs",
                        help='학습 로그 저장 디렉토리')
    parser.add_argument('--final_model_dir', type=str, default="final_model",
                        help='최종 학습된 모델과 설정 파일 저장 디렉토리')
    
    # DDP 관련 (torchrun/torch.distributed.launch에서 자동 설정됨)
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0),
                        help='DDP를 위한 로컬 프로세스 순위 (자동 설정)')
    
    # WandB 관련
    parser.add_argument('--use_wandb', action='store_true',
                        help='Weights & Biases 로깅 사용 여부')
    parser.add_argument('--wandb_project', type=str, default="segmentation-project",
                        help='WandB 프로젝트 이름')

    # DDP 마스터 주소 및 포트 (필요 시 수정)
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '12355'))

    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_args()

    # DDP를 위한 world_size 및 rank 설정
    # torchrun 사용 시 자동으로 환경 변수 설정됨
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.rank = int(os.getenv('RANK', 0))

    # 로깅 설정 (기본)
    setup_logging(args.rank, args.log_dir)

    if args.run_hpo:
        # HPO는 복잡성을 줄이기 위해 단일 GPU(rank 0)에서만 실행
        if args.rank == 0:
            logging.info("Starting Hyperparameter Optimization...")
            best_params = run_hyperparameter_optimization(args)
            if best_params:
                logging.info(f"HPO finished. Best parameters found: {best_params}")
                # HPO 완료 후 자동으로 최종 학습을 시작하려면 아래 주석 해제
                # if args.run_final_training:
                #     train_final_model(best_params, args)
        else:
            logging.info(f"Rank {args.rank} is idle during HPO.")

    elif args.run_final_training:
        best_params = None
        if args.rank == 0:
            logging.info("Starting Final Model Training...")
            try:
                with open(args.best_params_path, 'r') as f:
                    best_params = json.load(f)
                logging.info(f"Loaded best parameters from {args.best_params_path}")
            except FileNotFoundError:
                logging.error(f"Best parameters file not found at {args.best_params_path}. "
                              "Please run HPO first or provide a valid path.")
                return

        # best_params 객체를 모든 프로세스에 브로드캐스트
        params_list = [best_params]
        if args.world_size > 1:
            dist.init_process_group(backend='nccl', init_method=f'env://', 
                                    world_size=args.world_size, rank=args.rank)
            dist.broadcast_object_list(params_list, src=0)
        
        best_params = params_list[0]
        if best_params:
            train_final_model(best_params, args)
        
        if args.world_size > 1:
            cleanup()
            
    else:
        if args.rank == 0:
            logging.warning("Please specify an action: --run_hpo or --run_final_training")


if __name__ == '__main__':
    # DDP 실행을 위해 torchrun 사용 권장
    # 예시:
    # 1. 하이퍼파라미터 최적화 (단일 GPU):
    # python hyperparameter_optimization.py --run_hpo --n_trials 50 --optuna_epochs 20
    #
    # 2. 최적 파라미터로 최종 모델 학습 (다중 GPU):
    # torchrun --nproc_per_node=2 hyperparameter_optimization.py --run_final_training --final_epochs 100 --best_params_path studies/best_params.json --use_wandb
    main()