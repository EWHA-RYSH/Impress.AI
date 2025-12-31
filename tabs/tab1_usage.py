# ======================================================
# Tab 1 — 콘텐츠 활용도 모니터링
# ======================================================

import streamlit as st
import pandas as pd

from utils.data_loader import load_meta_df
from utils.eda_metrics import preprocess_country_data, get_image_type_distribution
from utils.charts import plot_image_type_distribution
from utils.insight_text import generate_usage_insights

def render():
    """콘텐츠 활용도 모니터링 탭 렌더링"""
    st.subheader("📊 콘텐츠 활용 모니터링")
    
    # 데이터 로드
    df_meta = load_meta_df()
    
    # 국가 선택 (사이드바에서 선택된 국가 사용)
    if "selected_country" in st.session_state:
        selected_country = st.session_state.selected_country
    else:
        countries = sorted(df_meta["country"].unique())
        selected_country = st.selectbox("국가 선택", countries, key="tab1_country")
    
    # 국가별 데이터 전처리
    df_country = preprocess_country_data(df_meta, selected_country)
    
    if len(df_country) == 0:
        st.warning(f"선택한 국가({selected_country})에 대한 데이터가 없습니다.")
        return
    
    st.info(f"📊 **{selected_country}** 시장: 총 {len(df_country)}개 게시글")
    
    # 이미지 타입별 분포
    st.markdown("---")
    st.markdown("### I. 이미지 타입별 활용 분포")
    
    type_count, type_ratio = get_image_type_distribution(df_country)
    
    # 인사이트 텍스트
    insights = generate_usage_insights(type_count, type_ratio, selected_country)
    st.markdown(insights)
    
    # 차트
    plot_image_type_distribution(type_count, type_ratio, selected_country)
    
    # 상세 통계 테이블
    with st.expander("📋 상세 통계"):
        summary_df = pd.DataFrame({
            "이미지 타입": type_count.index,
            "개수": type_count.values,
            "비율": [f"{ratio*100:.2f}%" for ratio in type_ratio.values]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
