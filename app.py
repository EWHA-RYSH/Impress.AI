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
import textwrap

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

@st.cache_data
def load_data():
    df = pd.read_excel("agent6_final_db.xlsx")
    return df

df = load_data()

countries = sorted(df["country"].unique())
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

def performance_level(ecdf: float):
    if ecdf >= 90:
        return "매우 높음", "badge-very-high"
    elif ecdf >= 75:
        return "높음", "badge-high"
    elif ecdf >= 50:
        return "보통", "badge-mid"
    elif ecdf >= 25:
        return "낮음", "badge-low"
    else:
        return "매우 낮음", "badge-very-low"


# ======================================================
# CSS
# ======================================================
CARD_CSS = """
<style>
/* 카드 래퍼 */
.result-card{
  background:#ffffff;
  border-radius:18px;
  padding:26px 26px 22px 26px;
  box-shadow:0 10px 28px rgba(0,0,0,.08);
  border:1px solid rgba(0,0,0,.06);
}

/* 타이틀 */
.result-title{
  margin:0 0 14px 0;
  font-size:34px;
  font-weight:800;
  letter-spacing:-0.6px;
}

/* 메타 문장 */
.meta{
  margin:0;
  color:rgba(0,0,0,.62);
  font-size:14px;
}

/* 물음표 툴팁 */
.helpq{
  display:inline-flex;
  width:18px;height:18px;
  border-radius:999px;
  align-items:center;justify-content:center;
  margin-left:6px;
  background:rgba(0,0,0,.08);
  color:rgba(0,0,0,.6);
  font-weight:800;
  font-size:12px;
  cursor:help;
}

/* 큰 숫자 */
.big{
  margin:10px 0 4px 0;
  font-size:52px;
  font-weight:900;
  letter-spacing:-1px;
}

/* 배지 */
.badge-very-high {
  background: #1f7a3f;
  color: white;
}

.badge-high {
  background: #52c41a;
  color: white;
}

.badge-mid {
  background: #faad14;
  color: #111;
}

.badge-low {
  background: #fa8c16;
  color: white;
}

.badge-very-low {
  background: #8c8c8c;
  color: white;
}
[class^="badge-"] {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 13px;
}

/* 작은 칩 */
.small-metric{
  display:flex;
  gap:8px;
  flex-wrap:wrap;
  margin-top:8px;
}
.metric-chip{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:6px 10px;
  border-radius:12px;
  background:rgba(0,0,0,.04);
  font-size:13px;
  color:rgba(0,0,0,.75);
}

/* 구분선 */
.divider{
  height:1px;
  background:rgba(0,0,0,.08);
  margin:16px 0;
}

/* 섹션 타이틀 */
.section-title{
  font-size:16px;
  font-weight:900;
  margin-bottom:6px;
}

/* 타입 pill */
.type-pill{
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  background:rgba(99,102,241,.12);
  color:rgba(67,56,202,1);
  font-weight:800;
  font-size:13px;
  margin-bottom:6px;
}

/* AI 박스 */
.ai-box{
  background:rgba(0,0,0,.035);
  border:1px solid rgba(0,0,0,.06);
  border-radius:14px;
  padding:12px 12px;
}

/* 주의 문구 */
.note{
  margin:12px 0 0 0;
  color:rgba(0,0,0,.55);
  font-size:13px;
}
.helpq {
  position: relative;
}

.helpq:hover::after {
  content: attr(title);
  position: absolute;
  bottom: 140%;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0,0,0,0.85);
  color: #fff;
  padding: 6px 10px;
  border-radius: 8px;
  font-size: 12px;
  white-space: nowrap;
  z-index: 999;
}

</style>
"""

st.markdown(CARD_CSS, unsafe_allow_html=True)

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





# -----------------------------
# 1. Sidebar (국가 선택)
# -----------------------------
st.sidebar.header("🔧 Filters")
selected_country = st.sidebar.selectbox(
    "Select Country",
    countries
)

df_country = df[df["country"] == selected_country].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"📊 Records: {len(df_country)} images"
)
# ======================================================
# 3. Tabs
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 콘텐츠 활용 모니터링",
    "🔥 콘텐츠 반응 & 성과 분석",
    "💹 전략적 개선 포인트",
    "🤖 AI 콘텐츠 성과 예측"
])
# ======================================================
# TAB 1 — 콘텐츠 활용 모니터링
# ======================================================
with tab1:
    st.subheader("📊 콘텐츠 활용 모니터링")
    st.caption("이 국가 계정에서 이미지 유형이 어떻게 활용되고 있는지 보여줍니다.")

    st.info("여기에 관련 그래프/요약 들어갈 자리")

# ======================================================
# TAB 2 — 콘텐츠 반응 & 성과 분석
# ======================================================
with tab2:
    st.subheader("🔥 콘텐츠 반응 & 성과 분석")
    st.caption("이미지 유형별 평균 성과와 고성과 진입 가능성을 함께 분석합니다.")

    st.info("여기에 관련 그래프/요약 들어갈 자리")

# ==================================================
# Tab 3 - 전략적 개선 포인트
# ==================================================
with tab3:
    st.subheader("💹 전략적 개선 포인트")
    st.caption("활용도와 성과를 비교하여 전략적 기회를 도출합니다.")

    st.info("Usage vs Performance / 과소·과대 활용 유형")


# ======================================================
# TAB 4 —  AI 콘텐츠 성과 예측
# ======================================================

with tab4:
    st.subheader("🤖 AI 콘텐츠 성과 예측")

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        uploaded = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])
        country = st.selectbox("국가 선택", country_list)

        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, width=360)

    if uploaded is not None:
        img_tensor = transform(image).unsqueeze(0)

        country_vec = country_encoder.transform(pd.DataFrame([[country]], columns=["country"]))
        country_vec = torch.tensor(country_vec, dtype=torch.float32)

        with torch.no_grad():
            cls_out, reg_out = model(img_tensor, country_vec)

        cls_idx = int(torch.argmax(cls_out, dim=1).item())
        img_type = cls_idx + 1

        type_name = TYPE_DESC.get(img_type, None)
        if type_name is None:
            type_name = f"유형 매핑 실패(예측값={img_type})"

        pred_z = float(reg_out.item())
        pred_logeng = pred_z * sigma + mu  # 모델이 예측한 log(1+eng_rate)

        ecdf = get_ecdf_percentile(df_ref, country, img_type, pred_logeng)
        percent = 50.0 if ecdf is None else ecdf

        level, badge_class = performance_level(percent)
        top10_msg = top10_badge(percent)

        tooltip = "동일 국가·유형의 과거 게시물 성과 분포(ECDF)에서, 이 이미지가 위치한 상대 백분위입니다."

        card_html = textwrap.dedent(f"""
        <div class="result-card">
          <h2 class="result-title">🔮 예측 결과</h2>

          <p class="meta">
            {country} 시장 기준 ‘상대 성과 위치(ECDF)’
            <span class="helpq" title="동일 국가·유형 콘텐츠 중 해당 이미지보다 성과가 낮은 비율">?</span>

          <div class="big">{percent:.1f}%</div>
          <span class="{badge_class}">{level}</span>

          <div class="small-metric">
            <div class="metric-chip"><b>예측 log-eng</b> : {pred_logeng:.4f}</div>
            <div class="metric-chip">{top10_msg}</div>
          </div>

          <div class="divider"></div>

          <div class="section-title">📌 이미지 유형</div>
          <p style="margin:0;">
            <span class="type-pill">Type {img_type}</span><br/>
            {type_name}
          </p>

          <div class="divider"></div>

          <div class="section-title">🧠 AI 해석</div>
          <div class="ai-box">
            <p style="margin:0 0 8px 0;">
              <span class="kicker">{country} 시장 기준</span>으로 이 이미지는 <b>Type {img_type}</b>로 분류되었습니다.
            </p>
            <p style="margin:0 0 8px 0;">
              예측 성과는 동일 국가·유형 콘텐츠 분포 대비 <b>{percent:.1f}%</b> 위치이며, 종합 레벨은 <b>{level}</b>입니다.
            </p>
            <p class="meta" style="margin:0;">
              (참고) log-eng는 <b>log(1 + eng_rate)</b> 형태의 예측값입니다.
            </p>
          </div>

          <p class="note">
            ※ 본 결과는 “절대 수치 예측”이 아니라, 동일 국가 내 콘텐츠 비교를 위한 “상대 지표”입니다.
          </p>
        </div>
        """)

        with right:
            st.markdown(card_html, unsafe_allow_html=True)

    else:
        with right:
            st.info("⬅️ 이미지를 업로드하면 예측 결과 카드가 표시됩니다.")