import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SurfaceDatasetPreprocessor:
    """
    XML 어노테이션을 읽어서 클래스별 인덱스 마스크로 변환하고,
    생성된 마스크 중 1개를 시각적으로 검증하는 기능을 포함한 전처리기
    """
    
    def __init__(self, root_dir: str, output_dir: str):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 검증용 이미지 저장 폴더 생성
        self.verification_dir = self.output_dir / 'verification'
        self.verification_dir.mkdir(exist_ok=True)
        self.verification_saved = False # 검증용 이미지가 저장되었는지 확인하는 플래그

        self.class_to_idx = {
            'background': 0, 'caution_zone': 1, 'bike_lane': 2,
            'alley': 3, 'roadway': 4, 'braille_guide_blocks': 5, 'sidewalk': 6
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # 클래스별 색상 (시각화용, RGB 순서)
        self.class_colors = {
            0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
            4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255),
        }
    
    def parse_xml_annotations(self, xml_path: str) -> Dict:
        """XML 파일에서 어노테이션 정보 추출"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        annotations = {}
        for image_elem in root.findall('.//image'):
            image_name = image_elem.get('name')
            annotations[image_name] = {
                'polygons': []
            }
            for polygon in image_elem.findall('polygon'):
                label = polygon.get('label')
                points_str = polygon.get('points')
                points = [tuple(map(int, map(float, p.split(',')))) for p in points_str.split(';')]
                annotations[image_name]['polygons'].append({'label': label, 'points': points})
        return annotations
    
    def create_mask_from_annotations(self, image_name: str, annotations: Dict, 
                                   width: int, height: int) -> np.ndarray:
        """
        어노테이션 정보로부터 클래스 인덱스 마스크 생성 (cv2.fillPoly 사용)
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if image_name in annotations:
            img_info = annotations[image_name]
            # 각 폴리곤을 클래스 인덱스로 채우기
            for polygon_info in img_info['polygons']:
                label = polygon_info['label']
                points = np.array(polygon_info['points'], dtype=np.int32)
                
                if label in self.class_to_idx:
                    class_idx = self.class_to_idx[label]
                    # cv2.fillPoly를 사용하여 NumPy 배열에 직접 폴리곤을 그립니다.
                    cv2.fillPoly(mask, [points], color=class_idx)
        
        return mask
    
    def process_surface_folder(self, surface_folder: str) -> List[Dict]:
        """개별 Surface 폴더 처리"""
        surface_path = self.root_dir / surface_folder
        xml_files = list(surface_path.glob('*.xml'))
        if not xml_files:
            logger.warning(f"XML 파일 없음: {surface_path}")
            return []
        
        annotations = self.parse_xml_annotations(str(xml_files[0]))
        processed_data = []
        
        image_files = sorted(list(surface_path.glob('MP_SEL_SUR_*.jpg')))

        for image_file in image_files:
            image_name = image_file.name
            image = cv2.imread(str(image_file))
            if image is None:
                logger.warning(f"이미지 로드 실패: {image_file}")
                continue
            
            height, width, _ = image.shape
            mask = self.create_mask_from_annotations(image_name, annotations, width, height)
            
            # --- 검증 단계: 첫 번째 유효한 마스크를 컬러 이미지로 저장 ---
            # 아직 검증 이미지를 저장하지 않았고, 마스크에 유효한 값이 있는 경우
            if not self.verification_saved and np.any(mask > 0):
                colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                for class_idx, color in self.class_colors.items():
                    # OpenCV는 BGR 순서를 사용하므로 RGB 색상을 변환 (color[::-1])
                    colored_mask[mask == class_idx] = color[::-1]
                
                # 원본 이미지와 컬러 마스크를 가로로 이어 붙여 비교 이미지 생성
                comparison_image = np.hstack([image, colored_mask])
                verification_path = self.verification_dir / f"VERIFICATION_{surface_folder}_{image_name}"
                cv2.imwrite(str(verification_path), comparison_image)
                logger.info(f"✅ 검증용 마스크 저장 완료: {verification_path}")
                self.verification_saved = True # 플래그를 올려서 다시 저장되지 않도록 함

            # --- 학습용 데이터 저장 (기존과 동일) ---
            output_img_path = self.output_dir / 'images' / f"{surface_folder}_{image_name}"
            output_mask_path = self.output_dir / 'masks' / f"{surface_folder}_{image_name.replace('.jpg', '.png')}"
            output_img_path.parent.mkdir(parents=True, exist_ok=True)
            output_mask_path.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(output_img_path), image)
            cv2.imwrite(str(output_mask_path), mask) # 학습에는 단일 채널 인덱스 마스크 사용
            
            processed_data.append({
                'image_path': str(output_img_path), 'mask_path': str(output_mask_path),
                'original_surface': surface_folder, 'original_image': image_name
            })
        return processed_data

    def process_all_surfaces(self) -> List[Dict]:
        """모든 Surface 폴더를 처리"""
        all_data = []
        surface_folders = sorted([d for d in os.listdir(self.root_dir) if d.startswith('Surface_') and os.path.isdir(self.root_dir / d)])
        logger.info(f"{self.root_dir} 에서 {len(surface_folders)}개의 Surface 폴더 발견")
        for surface_folder in surface_folders:
            logger.info(f"폴더 처리 중: {surface_folder}...")
            folder_data = self.process_surface_folder(surface_folder)
            all_data.extend(folder_data)
        return all_data


if __name__ == "__main__":
    # --- 1. 여러 디렉터리에서 데이터 전처리 및 수집 ---
    output_dir = Path("./processed_dataset")
    root_dirs = [f"./surface{i}" for i in range(1, 7)]

    all_processed_data = []
    # 전체 실행 과정에서 검증 이미지가 단 한 번만 저장되도록 상태를 관리
    verification_already_saved = False

    logger.info(f"다음 디렉터리들에 대한 처리를 시작합니다: {root_dirs}")

    for directory in root_dirs:
        if not os.path.exists(directory):
            logger.warning(f"디렉터리를 찾을 수 없습니다: {directory}")
            continue

        logger.info(f"--- 메인 디렉터리 처리 중: {directory} ---")
        preprocessor = SurfaceDatasetPreprocessor(
            root_dir=directory,
            output_dir=str(output_dir)
        )
        # 이전 디렉터리에서 검증 이미지를 저장했다면, 플래그를 전달
        preprocessor.verification_saved = verification_already_saved
        
        processed_data = preprocessor.process_all_surfaces()
        all_processed_data.extend(processed_data)

        # 현재 preprocessor의 상태를 전역 상태에 업데이트
        if preprocessor.verification_saved:
            verification_already_saved = True

    # --- 2. 수집된 전체 데이터 분할 및 메타데이터 저장 ---
    if not all_processed_data:
        logger.error("처리된 데이터가 없습니다. 프로그램을 종료합니다.")
    else:
        logger.info("--- 최종 메타데이터 집계 및 저장 ---")
        random.shuffle(all_processed_data)
        split_idx = int(len(all_processed_data) * 0.8)
        train_data = all_processed_data[:split_idx]
        valid_data = all_processed_data[split_idx:]

        # 메타데이터 저장을 위해 preprocessor 인스턴스 정보 사용
        temp_preprocessor = SurfaceDatasetPreprocessor(root_dir=".", output_dir=str(output_dir))
        metadata = {
            'num_classes': temp_preprocessor.num_classes, 'class_to_idx': temp_preprocessor.class_to_idx,
            'idx_to_class': temp_preprocessor.idx_to_class, 'class_colors': temp_preprocessor.class_colors,
            'train_data': train_data, 'valid_data': valid_data
        }
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("모든 디렉터리 처리 완료.")
        logger.info(f"총 처리된 이미지 수: {len(all_processed_data)}")
        logger.info(f"학습 데이터: {len(train_data)}, 검증 데이터: {len(valid_data)}")
        logger.info(f"메타데이터 저장 경로: {metadata_path}")