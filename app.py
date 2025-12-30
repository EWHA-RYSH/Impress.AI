# ======================================================
# Impress.AI — Final Streamlit App
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import os
from streamlit.components.v1 import html

# ======================================================
# Page Config
# ======================================================
st.set_page_config(
    page_title="Impress.AI",
    page_icon="📸",
    layout="wide"
)

# ======================================================
# Header
# ======================================================
st.markdown(
    """
    <div style="text-align:center; margin-bottom: 30px;">
        <h1 style="font-size:48px; font-weight:800;">
            Impress<span style="color:#3b82f6;">.AI</span>
        </h1>
        <p style="font-size:18px; color:#6b7280;">
            Image-based Content Performance Insight
        </p>
    </div>
    <hr style="border:none; height:1px; background-color:#e5e7eb; margin-bottom:30px;">
    """,
    unsafe_allow_html=True
)


# ======================================================
# Load Reference Data (for ECDF)
# ======================================================
@st.cache_data
def load_reference_df():
    df = pd.read_excel("agent6_final_reg_db.xlsx")
    df["log_eng"] = np.log1p(df["eng_rate"])
    return df

df_ref = load_reference_df()

@st.cache_data
def load_meta_df():
    return pd.read_excel("agent6_final_db.xlsx")

df_meta = load_meta_df()
countries = sorted(df_meta["country"].unique())

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
    with open("country_encoder.pkl", "rb") as f:
        country_encoder = pickle.load(f)

    with open("logengZ_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = MultiTaskModel(num_country=len(country_encoder.categories_[0]))
    model.load_state_dict(
        torch.load("final_multitask_logengZ_model.pth", map_location="cpu")
    )
    model.eval()

    return model, country_encoder, scaler["mu"], scaler["sigma"]

model, country_encoder, mu, sigma = load_model_bundle()
country_list = list(country_encoder.categories_[0])

# ======================================================
# Image Transform
# ======================================================
transform = transforms.Compose([
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

def performance_level(ecdf):
    if ecdf >= 80:
        return "높음", "badge-high"
    elif ecdf >= 50:
        return "보통", "badge-mid"
    else:
        return "낮음", "badge-low"

# ======================================================
# ECDF (Country-level)
# ======================================================
def get_country_ecdf_percentile(df_ref, country, pred_logeng):
    ref = df_ref[df_ref["country"] == country]["log_eng"].dropna().values
    if len(ref) == 0:
        return 50.0
    return float((ref < pred_logeng).mean() * 100.0)

# ======================================================
# Sidebar
# ======================================================
st.sidebar.header("🔧 Filters")
selected_country = st.sidebar.selectbox("Select Country", countries)
st.sidebar.caption(
    f"📊 Records: {len(df_meta[df_meta['country']==selected_country])}"
)

# ======================================================
# Tabs
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "📊 콘텐츠 활용 모니터링",
    "🔥 콘텐츠 성과 분석 & 패턴 도출",
    "🤖 AI 콘텐츠 성과 예측"
])

# ======================================================
# TAB 1
# ======================================================
with tab1:
    st.subheader("📊 콘텐츠 활용 모니터링")
    st.info("이 영역은 추후 시각화 추가 예정")

# ======================================================
# TAB 2
# ======================================================
with tab2:
    st.subheader("🔥 콘텐츠 성과 분석 & 패턴 도출")
    st.info("이 영역은 추후 고성과 패턴 분석 추가 예정")

# ======================================================
# TAB 3 — Prediction
# ======================================================
with tab3:
    st.subheader("🤖 AI 콘텐츠 성과 예측")

    left, right = st.columns([1, 1.4])

    with left:
        uploaded = st.file_uploader(
            "이미지 업로드",
            type=["jpg", "jpeg", "png"]
        )
        country = st.selectbox("국가 선택", country_list)

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_column_width=True)

            img_tensor = transform(image).unsqueeze(0)

            country_vec = country_encoder.transform(
                pd.DataFrame([[country]], columns=["country"])
            )
            country_vec = torch.tensor(country_vec, dtype=torch.float32)

            with torch.no_grad():
                cls_out, reg_out = model(img_tensor, country_vec)

            img_type = int(torch.argmax(cls_out, dim=1).item()) + 1
            pred_z = float(reg_out.item())
            pred_logeng = pred_z * sigma + mu
            percent = get_country_ecdf_percentile(df_ref, country, pred_logeng)

            type_name = TYPE_DESC.get(img_type, f"Type {img_type}")
            level, badge_class = performance_level(percent)
            
            card_html = f"""
            <style>
            .result-card {{
            background: #ffffff;
            padding: 28px;
            border-radius: 20px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 10px 24px rgba(0,0,0,0.06);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", sans-serif;
            }}

            .badge-high, .badge-mid, .badge-low {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 14px;
            }}

            .badge-high {{ background: #dcfce7; color: #166534; }}
            .badge-mid  {{ background: #fef9c3; color: #854d0e; }}
            .badge-low  {{ background: #fee2e2; color: #991b1b; }}

            .muted {{ color:#6b7280; }}
            .hr {{ border:none; height:1px; background:#e5e7eb; margin:18px 0; }}
            .h1 {{ font-size: 40px; margin: 8px 0 6px; font-weight: 800; }}
            .h2 {{ font-size: 26px; margin: 0 0 6px; font-weight: 800; }}
            .h4 {{ font-size: 16px; margin: 14px 0 6px; font-weight: 800; }}
            .small {{ color:#6b7280; font-size:13px; line-height:1.45; }}
            </style>

            <div class="result-card">
            <div class="h2">🔮 예측 결과</div>
            <div class="muted">{country} 시장 내 전체 콘텐츠 대비 예상 위치</div>

            <div class="h1">{percent:.1f}%</div>
            <span class="{badge_class}">{level}</span>

            <div class="hr"></div>

            <div class="h4">📌 이미지 유형</div>
            <div><b>Type {img_type}</b> · {type_name}</div>

            <div class="h4">🧠 AI 해석</div>
            <div style="line-height:1.55;">
                이 이미지는 <b>{country} 시장 기준</b>으로,
                전체 콘텐츠 분포 대비 <b>{level}</b> 수준의
                상대적 성과 위치에 해당합니다.
            </div>

            <div style="margin-top:10px;" class="small">
                ※ 본 결과는 절대적인 반응 수치가 아닌,
                동일 국가 내 콘텐츠 간 상대적 위치(percentile)를 의미합니다.
            </div>
            </div>
            """

            with right:
                html(card_html, height=430)


        else:
            with right:
                st.info("⬅️ 이미지를 업로드하면 예측 결과가 표시됩니다.")
