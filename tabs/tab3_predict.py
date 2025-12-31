# ======================================================
# Tab 3 — CV 기반 성과 예측
# ======================================================

import streamlit as st
import pandas as pd
import torch
from PIL import Image
from streamlit.components.v1 import html

from models.cv_model import load_model_bundle, get_image_transform, TYPE_DESC
from utils.eda_metrics import get_country_ecdf_percentile, performance_level
from components.style import inject_style

def render(df_ref):
    """AI 콘텐츠 성과 예측 탭 렌더링"""
    st.subheader("🤖 AI 콘텐츠 성과 예측")

    # 모델 로드
    model, country_encoder, mu, sigma = load_model_bundle()
    country_list = list(country_encoder.categories_[0])
    transform = get_image_transform()

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
            
            # 스타일 주입
            inject_style()
            
            card_html = f"""
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

