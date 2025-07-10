import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
import inspect # forward 메소드 소스 코드를 확인하기 위해 import

# 이 코드를 실행하기 전에 surface_dataset.py가 같은 디렉토리에 있거나
# 파이썬 경로에 포함되어 있어야 합니다.
from surface_dataset import SurfaceSegmentationDataset

# --- 0. 환경 설정 ---
# 사용 가능한 경우 GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔄 Using device: {device}")

# --- 1. 데이터셋 및 데이터로더 준비 ---
print("\n🔄 1. 데이터셋 및 데이터로더 준비")
metadata_path = "/home/work/data/indo_walking/surface_masking/processed_dataset/metadata.json"
data_base_path = os.path.dirname(os.path.dirname(metadata_path))

with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 데이터셋 인스턴스 생성
train_dataset = SurfaceSegmentationDataset(metadata['train_data'], target_size=(512, 512), is_train=True, data_base_path=data_base_path)

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2) # 배치 사이즈는 GPU 메모리에 맞게 조절

# --- 2. 모델 준비 ---
print("\n🔄 2. 모델 준비")
# 사전 학습된 Segformer 모델 로드
pretrained_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
pretrained_model_name = "nvidia/mit-b0"
model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name)

# --- 클래스 개수 변경 로직 ---
# 데이터셋의 클래스 개수에 맞게 모델의 최종 분류 레이어를 교체
new_num_classes = 7 # 데이터셋에서 클래스 개수 가져오기
decoder_hidden_size = model.decode_head.classifier.in_channels
model.decode_head.classifier = nn.Conv2d(decoder_hidden_size, new_num_classes, kernel_size=1)
model.config.num_labels = new_num_classes # 모델 설정도 업데이트

model.to(device)
model.eval()  # 평가 모드

print(f"모델: {pretrained_model_name}")
print(f"수정된 클래스 개수: {model.config.num_labels}")
print("-" * 50)


# --- 3. 실제 데이터로 Loss 비교 ---
print("🚀 3. 실제 데이터로 Loss 비교 시작")

# 데이터로더에서 데이터 한 배치 가져오기
try:
    batch = next(iter(train_loader))
    pixel_values = batch['pixel_values'].to(device)
    labels = batch['labels'].to(device)
    print(f"입력 이미지 shape: {pixel_values.shape}")
    print(f"정답 레이블 shape: {labels.shape}")
    print("-" * 50)

    # 3.1. 자동 계산 (Hugging Face 내장 방식)
    print("🚀 3.1. Hugging Face 내장 방식으로 Loss 자동 계산")
    with torch.no_grad():
        # labels를 함께 전달하면 loss가 자동으로 계산됨
        outputs_auto = model(pixel_values=pixel_values, labels=labels)

    auto_loss = outputs_auto.loss
    auto_logits = outputs_auto.logits

    print(f"자동 계산된 Loss: {auto_loss.item():.6f}")
    print(f"자동 계산 시 반환된 logits shape: {auto_logits.shape}")
    print("-" * 50)

    # 3.2. 수동 계산 (내부 로직 재현)
    print("🛠️ 3.2. 내부 로직을 재현하여 Loss 수동 계산")

    # Logits 생성 (labels 없이 전달)
    with torch.no_grad():
        outputs_manual = model(pixel_values=pixel_values)
    manual_logits = outputs_manual.logits

    # Logits 업샘플링
    upsampled_logits = F.interpolate(
        manual_logits,
        size=labels.shape[-2:], # 레이블의 H, W 크기에 맞춤
        mode='bilinear',
        align_corners=False
    )

    # CrossEntropyLoss 계산
    # surface_dataset.py 구현에 따라 ignore_index가 필요할 수 있음
    # 만약 특정 값(예: 255)을 무시해야 한다면 아래와 같이 설정
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=255) 
    manual_loss = loss_fct(upsampled_logits, labels)

    print(f"수동 계산된 Loss: {manual_loss.item():.6f}")
    print("-" * 50)

    # 3.3. 결과 비교
    print("✅ 3.3. 최종 결과 비교")
    are_losses_close = torch.allclose(auto_loss, manual_loss)
    print(f"자동 계산 Loss와 수동 계산 Loss가 일치하는가? -> {are_losses_close}")

    if are_losses_close:
        print("\n🎉 성공: 실제 데이터셋에서도 모델의 내부 Loss 계산 과정을 정확하게 재현했습니다.")
    else:
        print("\n❌ 실패: 자동 계산과 수동 계산 Loss가 일치하지 않습니다. ignore_index나 데이터 타입을 확인해보세요.")

except StopIteration:
    print("데이터로더에서 데이터를 가져올 수 없습니다. 데이터셋이 비어있는지 확인하세요.")

breakpoint()

import inspect

# 모델의 forward 메소드 소스 코드를 직접 출력합니다.
print(inspect.getsource(model.forward))



# (Pdb) print(inspect.getsource(model.forward))
#     @auto_docstring
#     def forward(
#         self,
#         pixel_values: torch.FloatTensor,
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[tuple, SemanticSegmenterOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
#             Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

#         Examples:

#         ```python
#         >>> from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
#         >>> from PIL import Image
#         >>> import requests

#         >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
#         >>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> inputs = image_processor(images=image, return_tensors="pt")
#         >>> outputs = model(**inputs)
#         >>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
#         >>> list(logits.shape)
#         [1, 150, 128, 128]
#         ```"""
        
"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if labels is not None and self.config.num_labels < 1:
            raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)
            elif self.config.num_labels == 1:
                valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                loss_fct = BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
                loss = (loss * valid_mask).mean()

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
        
"""