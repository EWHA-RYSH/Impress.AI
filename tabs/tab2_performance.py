# ======================================================
# Tab 2 — 성과 분석 & 패턴 도출
# ======================================================

import streamlit as st
import pandas as pd

from utils.data_loader import load_meta_df
from utils.eda_metrics import (
    preprocess_country_data,
    get_image_type_distribution,
    get_performance_summary,
    get_top_percentile_metrics,
    get_stability_metrics,
    get_usage_vs_performance,
    get_response_characteristics
)
from utils.charts import (
    plot_performance_comparison,
    plot_performance_summary,
    plot_top_percentile_probability,
    plot_top_percentile_concentration,
    plot_stability_metrics,
    plot_usage_vs_performance,
    plot_likes_vs_comments,
    plot_comment_ratio
)
from utils.insight_text import (
    generate_performance_insights,
    generate_top_percentile_insights,
    generate_stability_insights,
    generate_strategy_insights,
    generate_summary_insights
)

def render():
    """성과 분석 & 패턴 도출 탭 렌더링"""
    st.subheader("🔥 콘텐츠 성과 분석 & 패턴 도출")
    
    # 데이터 로드
    df_meta = load_meta_df()
    
    # 국가 선택
    if "selected_country" in st.session_state:
        selected_country = st.session_state.selected_country
    else:
        countries = sorted(df_meta["country"].unique())
        selected_country = st.selectbox("국가 선택", countries, key="tab2_country")
    
    # 국가별 데이터 전처리
    df_country = preprocess_country_data(df_meta, selected_country)
    
    if len(df_country) == 0:
        st.warning(f"선택한 국가({selected_country})에 대한 데이터가 없습니다.")
        return
    
    st.info(f"📊 **{selected_country}** 시장: 총 {len(df_country)}개 게시글")
    
    # 탭으로 섹션 구분
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 성과 비교",
        "🏆 고성과 분석",
        "📊 안정성 분석",
        "💬 반응 성격",
        "🎯 전략 인사이트"
    ])
    
    # ==========================================
    # Tab 1: 성과 비교
    # ==========================================
    with tab1:
        st.markdown("### II. 이미지 타입별 성과 비교")
        
        # 성과 요약
        agg_perf = get_performance_summary(df_country)
        
        # 인사이트
        insights = generate_performance_insights(agg_perf, selected_country)
        st.markdown(insights)
        
        # 차트
        st.markdown("#### 참여율 (Engagement Rate) 분포")
        plot_performance_comparison(df_country, selected_country, "eng_rate")
        
        st.markdown("#### 평균 참여율")
        plot_performance_summary(agg_perf, selected_country, "eng_mean")
        
        st.markdown("#### 중앙값 참여율")
        plot_performance_summary(agg_perf, selected_country, "eng_median")
        
        st.markdown("#### 좋아요 분포")
        plot_performance_comparison(df_country, selected_country, "likes")
        
        st.markdown("#### 댓글 분포")
        plot_performance_comparison(df_country, selected_country, "comments")
        
        # 상세 통계
        with st.expander("📋 상세 성과 통계"):
            st.dataframe(agg_perf, use_container_width=True, hide_index=True)
    
    # ==========================================
    # Tab 2: 고성과 분석
    # ==========================================
    with tab2:
        st.markdown("### III. 고성과 콘텐츠 분석")
        
        # Top 10% 분석
        st.markdown("#### Top 10% 성과 분석")
        prob_10, conc_10, threshold_10 = get_top_percentile_metrics(df_country, 10)
        
        # 인사이트
        insights = generate_top_percentile_insights(prob_10, conc_10, selected_country, 10)
        st.markdown(insights)
        
        # 차트
        if len(prob_10) > 0:
            plot_top_percentile_probability(prob_10, selected_country, 10)
        
        if len(conc_10) > 0:
            plot_top_percentile_concentration(conc_10, selected_country, 10)
        
        st.caption(f"💡 Top 10% 기준선: 참여율 {threshold_10:.6f} 이상")
        
        # Top 30% 분석
        st.markdown("---")
        st.markdown("#### Top 30% 성과 분석")
        prob_30, conc_30, threshold_30 = get_top_percentile_metrics(df_country, 30)
        
        if len(prob_30) > 0:
            plot_top_percentile_probability(prob_30, selected_country, 30)
        
        if len(conc_30) > 0:
            plot_top_percentile_concentration(conc_30, selected_country, 30)
        
        st.caption(f"💡 Top 30% 기준선: 참여율 {threshold_30:.6f} 이상")
        
        # 상세 통계
        with st.expander("📋 Top 10%/30% 상세 통계"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 10% 확률**")
                st.dataframe(prob_10, use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**Top 10% 내 구성비**")
                st.dataframe(conc_10, use_container_width=True, hide_index=True)
    
    # ==========================================
    # Tab 3: 안정성 분석
    # ==========================================
    with tab3:
        st.markdown("### IV. 성과 안정성 분석")
        
        # 안정성 지표 계산
        stability = get_stability_metrics(df_country)
        
        # 인사이트
        insights = generate_stability_insights(stability, selected_country)
        st.markdown(insights)
        
        # 차트
        st.markdown("#### 표준편차 (STD)")
        plot_stability_metrics(stability, selected_country, "eng_std")
        
        st.markdown("#### IQR (사분위수 범위)")
        plot_stability_metrics(stability, selected_country, "eng_iqr")
        
        st.markdown("#### 변동계수 (CV)")
        plot_stability_metrics(stability, selected_country, "eng_cv")
        
        # 상세 통계
        with st.expander("📋 안정성 상세 통계"):
            st.dataframe(stability, use_container_width=True, hide_index=True)
    
    # ==========================================
    # Tab 4: 반응 성격
    # ==========================================
    with tab4:
        st.markdown("### V. 반응 성격 분석")
        
        # 반응 성격 분석
        comp = get_response_characteristics(df_country)
        
        # 차트
        st.markdown("#### 좋아요 vs 댓글")
        plot_likes_vs_comments(df_country, selected_country)
        
        st.markdown("#### 댓글 비율 분포")
        plot_comment_ratio(df_country, selected_country)
        
        # 상세 통계
        with st.expander("📋 반응 성격 상세 통계"):
            st.dataframe(comp, use_container_width=True, hide_index=True)
    
    # ==========================================
    # Tab 5: 전략 인사이트
    # ==========================================
    with tab5:
        st.markdown("### VI. 활용도 대비 전략적 개선 포인트")
        
        # 활용도 vs 성과 분석
        merged, underused, overused = get_usage_vs_performance(df_country, 10)
        
        # 인사이트
        insights = generate_strategy_insights(underused, overused, selected_country)
        st.markdown(insights)
        
        # 차트
        st.markdown("#### 활용도 vs 평균 성과")
        plot_usage_vs_performance(merged, selected_country, "eng_mean")
        
        st.markdown("#### 활용도 vs Top 10% 확률")
        plot_usage_vs_performance(merged, selected_country, "p_top10")
        
        # 상세 통계
        with st.expander("📋 활용도 vs 성과 상세 통계"):
            st.dataframe(merged, use_container_width=True, hide_index=True)
        
        # 최종 요약
        st.markdown("---")
        all_metrics = {
            "type_distribution": get_image_type_distribution(df_country),
            "performance_summary": agg_perf,
            "top10_prob": prob_10
        }
        summary = generate_summary_insights(df_country, selected_country, all_metrics)
        st.markdown(summary)
