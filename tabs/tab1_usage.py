import streamlit as st
import pandas as pd

from utils.data_loader import load_meta_df
from utils.eda_metrics import preprocess_country_data, get_image_type_distribution
from utils.metrics import compute_usage_kpis, format_percentage, format_engagement_rate
from utils.charts import plot_usage_distribution, plot_engagement_distribution
from components.layout import (
    render_page_header,
    render_kpi_card,
    render_insight_box,
    get_type_name,
    render_image_type_guide,
    section_gap
)

def render():
    df_meta = load_meta_df()
    selected_country = st.session_state.get("selected_country", sorted(df_meta["country"].unique())[0])
    df_country = preprocess_country_data(df_meta, selected_country)
    
    if len(df_country) == 0:
        st.warning(f"{selected_country}에 대한 데이터가 없습니다.")
        return
    
    countries = sorted(df_meta["country"].unique())
    render_page_header(
        "활용도 모니터링",
        countries=countries,
        selected_country=selected_country,
        n_posts=len(df_country),
        description="이미지 유형별 활용 빈도와 좋아요・댓글・참여율 분포를 함께 비교해 운영 방향을 도출합니다."
    )
    
    current_country = st.session_state.get("selected_country", selected_country)
    if current_country != selected_country:
        selected_country = current_country
        df_country = preprocess_country_data(df_meta, selected_country)
        if len(df_country) == 0:
            st.warning(f"{selected_country}에 대한 데이터가 없습니다.")
            return
    
    section_gap(16)
    with st.expander("📁 이미지 유형 기준", expanded=False):
        st.markdown(
            """
            <div style="
                font-size: 14px;
                color: #6B7280;
                line-height: 1.6;
                margin-bottom: 20px;
            ">
                Type 1~6은 게시물의 이미지 구성 방식이며, KPI 해석/성과 비교의 기준으로 사용됩니다.<br>
            </div>
            """,
            unsafe_allow_html=True
        )
        render_image_type_guide()
    
    section_gap(48)
    
    kpis = compute_usage_kpis(df_country)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_used_name = get_type_name(kpis['most_used']['type'])
        render_kpi_card(
            "가장 많이 사용된 타입",
            f"{most_used_name}",
            subtext=f"Type {kpis['most_used']['type']} · 전체의 {format_percentage(kpis['most_used']['pct'])}",
            highlight=True
        )
    
    with col2:
        least_used_name = get_type_name(kpis['least_used']['type'])
        render_kpi_card(
            "가장 적게 사용된 타입",
            f"{least_used_name}",
            subtext=f"Type {kpis['least_used']['type']} · 전체의 {format_percentage(kpis['least_used']['pct'])}"
        )
    
    with col3:
        if kpis['engagement_leader']['type']:
            leader_name = get_type_name(kpis['engagement_leader']['type'])
            render_kpi_card(
                "참여율 최고 타입",
                f"{leader_name}",
                subtext=f"Type {kpis['engagement_leader']['type']} · 참여율: {format_engagement_rate(kpis['engagement_leader']['value'])}"
            )
        else:
            render_kpi_card("참여율 최고 타입", "N/A")
    
    section_gap(48)
    
    type_count, type_ratio = get_image_type_distribution(df_country)
    
    st.markdown(
        """
        <div class="section">
            <h4 class="section-title">활용 분포</h4>
            <div class="section-desc">국가 계정에서 게시된 콘텐츠를 이미지 유형별로 분류하여,
각 이미지 타입이 전체 콘텐츠에서 차지하는 사용 비중을 확인합니다.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    section_gap(16)
    plot_usage_distribution(type_ratio, selected_country, highlight_type=kpis['most_used']['type'])
    
    section_gap(48)
    
    st.markdown(
        """
        <div class="section">
            <h4 class="section-title">참여율 분포</h4>
            <div class="section-desc">이미지 타입별 참여율의 분포를 비교하고,
유형별 반응 수준과 변동 폭을 함께 확인합니다.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    section_gap(16)
    plot_engagement_distribution(df_country, selected_country, highlight_type=kpis['engagement_leader']['type'])
    
    section_gap(48)
    
    insights = [
        f"<strong>{get_type_name(kpis['most_used']['type'])} (Type {kpis['most_used']['type']})</strong>가 가장 많이 사용되는 콘텐츠 타입으로, 전체의 {format_percentage(kpis['most_used']['pct'])}를 차지합니다.",
        f"<strong>{get_type_name(kpis['least_used']['type'])} (Type {kpis['least_used']['type']})</strong>는 활용도가 낮아 {format_percentage(kpis['least_used']['pct'])}만 사용되고 있습니다.",
    ]
    
    if kpis['engagement_leader']['type']:
        insights.append(
            f"<strong>{get_type_name(kpis['engagement_leader']['type'])} (Type {kpis['engagement_leader']['type']})</strong>는 평균 참여율이 가장 높아 활용도 증가 여지가 있습니다."
        )
    
    render_insight_box(insights)
    
    section_gap(48)
    
    with st.expander("상세 통계 보기", expanded=False):
        st.markdown(
            """
            <div style="
                font-size: 13px;
                color: #6B7280;
                line-height: 1.6;
                margin-bottom: 20px;
                font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
            ">
                타입별 기본 통계를 요약합니다.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        summary_df = pd.DataFrame({
            "이미지 타입": type_count.index,
            "개수": type_count.values,
            "비율": [f"{ratio*100:.2f}%" for ratio in type_ratio.values]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
