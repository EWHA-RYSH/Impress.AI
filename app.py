# ======================================================
# Impress.AI — Final App (Tab-based)
# Tab 3: CV Model + ECDF Relative Performance
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle

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
# Utility Functions
# ======================================================
def get_ecdf_percentile(df, country, img_type, pred_logeng):
    ref = df[
        (df["country"] == country) &
        (df["img_type"] == img_type)
    ]["log_eng"].values

    if len(ref) < 5:
        return None

    return (ref < pred_logeng).mean() * 100


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

    usage = df_view.groupby("img_type").size().reset_index(name="count")

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
        sns.boxplot(data=df_view, x="img_type", y="eng_rate", ax=ax)
        ax.set_yscale("log")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_view, x="img_type", y="eng_rank_country_type", ax=ax)
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

    if uploaded:
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

        ecdf = get_ecdf_percentile(
            df_ref, country, img_type, pred_logeng
        )

        with right:
            st.markdown("### 🔮 예측 결과")

            st.write(f"**예측 이미지 유형:** Type {img_type}")
            st.write(f"**예측 log-eng score:** {pred_logeng:.4f}")

            if ecdf is None:
                st.warning("기준 데이터가 부족하여 상대 성과를 계산할 수 없습니다.")
            else:
                st.metric(
                    label="상대 성과 위치 (ECDF)",
                    value=f"{ecdf:.1f}%",
                    help="동일 국가·유형 콘텐츠 중 해당 이미지보다 성과가 낮은 비율"
                )

                st.write(
                    f"👉 동일 국가·유형 콘텐츠 중 "
                    f"**약 {ecdf:.1f}%보다 높은 성과**가 예측됩니다."
                )

                st.markdown(f"### {top10_badge(ecdf)}")

                st.caption(
                    "※ 본 지표는 경험적 분포(ECDF)를 기반으로 한 상대 성과 평가입니다."
                )
    else:
        st.info("⬅️ 이미지를 업로드하면 예측 결과가 표시됩니다.")
