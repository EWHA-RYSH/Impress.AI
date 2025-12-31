# ======================================================
# CV Model — 모델 정의 및 로드
# ======================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import pickle
import os

# 모델 파일 경로
MODELS_DIR = os.path.join(os.path.dirname(__file__))

# ======================================================
# Model Definition (MUST MATCH TRAINING)
# ======================================================
class MultiTaskModel(nn.Module):
    def __init__(self, num_country):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.shared = nn.Sequential(
            nn.Linear(feat_dim + num_country, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.cls_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.reg_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.cls_head = nn.Linear(256, 6)
        self.reg_head = nn.Linear(256, 1)

    def forward(self, image, country_vec):
        feat = self.backbone(image)
        x = torch.cat([feat, country_vec], dim=1)
        x = self.shared(x)
        cls_out = self.cls_head(self.cls_branch(x))
        reg_out = self.reg_head(self.reg_branch(x)).squeeze(1)
        return cls_out, reg_out

# ======================================================
# Load Model Bundle
# ======================================================
@st.cache_resource
def load_model_bundle():
    """모델, 인코더, 스케일러 로드"""
    encoder_path = os.path.join(MODELS_DIR, "country_encoder.pkl")
    scaler_path = os.path.join(MODELS_DIR, "logengZ_scaler.pkl")
    model_path = os.path.join(MODELS_DIR, "final_multitask_logengZ_model.pth")
    
    with open(encoder_path, "rb") as f:
        country_encoder = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = MultiTaskModel(num_country=len(country_encoder.categories_[0]))
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()

    return model, country_encoder, scaler["mu"], scaler["sigma"]

# ======================================================
# Image Transform
# ======================================================
def get_image_transform():
    """이미지 전처리 변환"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ======================================================
# Constants
# ======================================================
TYPE_DESC = {
    1: "여러 제품을 함께 보여주는 제품 단체샷",
    2: "한 제품을 단독으로 강조한 제품 단독샷",
    3: "제품 제형/텍스처를 중심으로 한 제품 질감샷",
    4: "모델과 제품을 함께 배치한 이미지",
    5: "제품 없이 모델 중심으로 연출된 이미지",
    6: "여러 인물과 제품을 함께 보여주는 이미지"
}

