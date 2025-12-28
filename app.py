# ======================================================
# Impress.AI — App
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# Page Config
# ======================================================
st.set_page_config(
    page_title="Impress.AI",
    page_icon="📸",
    layout="wide"
)

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
st.markdown("""
<style>
/* Card */
.result-card{
  background:#ffffff;
  padding:28px;
  border-radius:20px;
  border:1px solid #e5e7eb;
  box-shadow:0 12px 30px rgba(0,0,0,0.08);
}
.result-title{
  font-size:32px;
  font-weight:800;
  margin:0 0 10px 0;
}
.result-subtitle{
  font-size:20px;
  font-weight:700;
  margin:0 0 6px 0;
}
.meta{
  color:#6b7280;
  font-size:13px;
  margin-top:8px;
}
.big{
  font-size:54px;
  font-weight:900;
  margin:10px 0 6px 0;
  letter-spacing:-0.5px;
}
.divider{
  height:1px;
  background:#e5e7eb;
  margin:18px 0;
}
.section-title{
  font-size:18px;
  font-weight:800;
  margin:0 0 8px 0;
}
.ai-box{
  background:#f9fafb;
  border:1px solid #e5e7eb;
  border-radius:16px;
  padding:16px;
}
.ai-box p{
  margin:10px 0;
  line-height:1.6;
}
.kicker{
  color:#111827;
  font-weight:800;
}
.note{
  color:#6b7280;
  font-size:12px;
  margin-top:10px;
}

/* Badges */
.badge{
  display:inline-block;
  padding:8px 16px;
  border-radius:999px;
  font-weight:800;
  font-size:14px;
}
.badge-high{ background:#dcfce7; color:#166534; }
.badge-mid{  background:#fef9c3; color:#854d0e; }
.badge-low{  background:#fee2e2; color:#991b1b; }

/* Type pill */
.type-pill{
  display:inline-block;
  padding:6px 12px;
  border-radius:999px;
  border:1px solid #e5e7eb;
  background:#f9fafb;
  font-weight:800;
}
</style>
""", unsafe_allow_html=True)


# ======================================================
# Load Reference Data
# ======================================================
@st.cache_data
def load_reference_df():
    df = pd.read_excel("agent6_final_reg_db.xlsx")
    df["log_eng"] = np.log1p(df["eng_rate"])
    return df

df_ref = load_reference_df()

# ======================================================
# Model Definition (must match training)
# ======================================================
class MultiTaskModel(nn.Module):
    def __init__(self, num_country, num_classes=6):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.fc_shared = nn.Sequential(
            nn.Linear(feat_dim + num_country, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.cls_head = nn.Linear(512, num_classes)
        self.reg_head = nn.Linear(512, 1)

    def forward(self, image, country_vec):
        feat = self.backbone(image)
        x = torch.cat([feat, country_vec], dim=1)
        x = self.fc_shared(x)
        return self.cls_head(x), self.reg_head(x).squeeze(1)

# ======================================================
# Load Model Bundle
# ======================================================
@st.cache_resource
def load_model_bundle():
    with open("country_encoder.pkl", "rb") as f:
        country_encoder = pickle.load(f)

    with open("logengZ_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    mu, sigma = scaler["mu"], scaler["sigma"]

    model = MultiTaskModel(
        num_country=len(country_encoder.categories_[0])
    )
    model.load_state_dict(
        torch.load("final_multitask_logengZ_model.pth", map_location="cpu")
    )
    model.eval()

    return model, country_encoder, mu, sigma

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

def performance_level(ecdf_percent: float):
    if ecdf_percent >= 80:
        return "높음", "badge badge-high"
    elif ecdf_percent >= 50:
        return "보통", "badge badge-mid"
    else:
        return "낮음", "badge badge-low"

    
# ======================================================
# Badge Style
# ======================================================
st.markdown("""
<style>
.badge-high {
    background:#dcfce7; color:#166534;
    padding:8px 18px; border-radius:999px;
    font-weight:700;
}
.badge-mid {
    background:#fef9c3; color:#854d0e;
    padding:8px 18px; border-radius:999px;
    font-weight:700;
}
.badge-low {
    background:#fee2e2; color:#991b1b;
    padding:8px 18px; border-radius:999px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# Utility Functions
# ======================================================
def get_country_ecdf_percentile(df_ref, country, pred_logeng):
    ref = df_ref[df_ref["country"] == country]["log_eng"].dropna().values

    if len(ref) == 0:
        return None

    return float((ref < pred_logeng).mean() * 100.0)


def top10_badge(ecdf):
    if ecdf >= 90:
        return "🔥 Top 10% 진입 가능성 높음"
    elif ecdf >= 80:
        return "⚡ Top 10% 진입 가능성 있음"
    else:
        return "ℹ️ Top 10% 진입 가능성 낮음"


# ======================================================
# Tabs
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "📊 활용도 모니터링",
    "🔥 반응 & 성과 분석",
    "🤖 CV 기반 콘텐츠 성과 예측"
])

# ======================================================
# TAB 1 — (기존 코드 유지 영역)
# ======================================================
with tab1:
    st.subheader("📊 이미지 유형 활용도")

    usage = df_ref.groupby("img_type").size().reset_index(name="count")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(usage, width=350)

    with col2:
        fig, ax = plt.subplots()
        sns.barplot(data=usage, x="img_type", y="count", ax=ax)
        st.pyplot(fig)

# ======================================================
# TAB 2 — (기존 코드 유지 영역)
# ======================================================
with tab2:
    st.subheader("🔥 이미지 유형별 반응 성과")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_ref, x="img_type", y="eng_rate", ax=ax)
        ax.set_yscale("log")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_ref, x="img_type", y="eng_rank_country_type", ax=ax)
        st.pyplot(fig)


# ======================================================
# TAB 3 — 🔥 FINAL MODEL
# ======================================================
with tab3:
    st.subheader("🤖 CV 기반 콘텐츠 성과 예측")

    left, right = st.columns([1, 1.4])

    with left:
        uploaded = st.file_uploader(
            "이미지 업로드",
            type=["jpg", "png", "jpeg"]
        )
        country = st.selectbox("국가 선택", country_list)
    
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image)
            img_tensor = transform(image).unsqueeze(0)

            country_vec = country_encoder.transform(
                pd.DataFrame([[country]], columns=["country"])
            )
            country_vec = torch.tensor(country_vec, dtype=torch.float32)

            with torch.no_grad():
                cls_out, reg_out = model(img_tensor, country_vec)

            cls_idx = int(torch.argmax(cls_out, dim=1).item())
            img_type = cls_idx + 1

    
            pred_z = float(reg_out.item())
            pred_logeng = pred_z * sigma + mu

        
            ecdf = get_country_ecdf_percentile(df_ref, country, pred_logeng)

            if ecdf is None or (isinstance(ecdf, float) and np.isnan(ecdf)):
                percent = 50.0
            else:
                percent = float(ecdf)

            
            type_name = TYPE_DESC.get(img_type, f"Type {img_type}")

            # 성과 수준(뱃지)
            level, badge_class = performance_level(percent)

            with right:
                st.markdown(f"""
                <div style="
                    background:#ffffff;
                    padding:28px;
                    border-radius:20px;
                    border:1px solid #e5e7eb;
                    box-shadow:0 10px 24px rgba(0,0,0,0.06);
                ">

                <h2 style="margin-bottom:6px;">🔮 예측 결과</h2>

                <p style="margin:0; color:#6b7280; font-size:13px;">
                    {country} 시장 내 콘텐츠 대비 예상 성과 위치
                </p>

                <h3 style="margin-top:8px; margin-bottom:6px;">
                    {percent:.1f}%
                </h3>

                <span class="{badge_class}" style="
                    display:inline-block;
                    padding:8px 18px;
                    border-radius:999px;
                    font-size:16px;
                    font-weight:700;
                    margin-top:4px;
                ">
                    {level}
                </span>

                <hr style="margin:22px 0;">

                <h4 style="margin-bottom:6px;">📌 이미지 유형</h4>
                <p style="margin-top:0;">
                    <b>Type {img_type}</b> · {type_name}
                </p>

                <h4 style="margin-bottom:8px;">🧠 AI 해석</h4>

                <p>
                    <b>{country} 시장 기준</b>으로 볼 때,
                    이 이미지는 <b>{type_name}</b> (Type {img_type}) 유형으로 분류되었습니다.
                </p>

                <p>
                    CV 모델이 예측한 반응 강도를
                    <b>{country} 시장 전체 콘텐츠 분포</b>와 비교한 결과,
                    <b>{level}</b> 수준의 상대적 성과 위치에 해당합니다.
                </p>

                <p style="color:#6b7280; font-size:13px;">
                    ※ 본 지표는 절대적인 성과 수치를 예측하기보다는,
                    동일 국가 내 콘텐츠 간 상대적 위치를 참고하기 위한 지표입니다.
                </p>

                </div>
                """, unsafe_allow_html=True)

        else:
            with right:
                st.info("⬅️ 이미지를 업로드하면 예측 결과가 표시됩니다.")
