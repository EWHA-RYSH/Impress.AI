# ======================================================
# Charts — 공통 차트 함수 (Streamlit용)
# ======================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 국가 코드 매핑
COUNTRY_NAMES = {
    "SG": "싱가포르",
    "MY": "말레이시아",
    "TH": "태국",
    "ID": "인도네시아",
    "PH": "필리핀",
    "VN": "베트남",
    "JP": "일본"
}

def get_country_name(code):
    """국가 코드를 한글 이름으로 변환"""
    return COUNTRY_NAMES.get(code, code)

def plot_image_type_distribution(type_count, type_ratio, country):
    """이미지 타입별 분포 차트"""
    country_name = get_country_name(country)
    
    fig = px.bar(
        x=type_ratio.index.astype(str),
        y=type_ratio.values,
        labels={"x": "이미지 타입", "y": "비율"},
        title=f"[{country_name}] 이미지 타입별 활용 비율"
    )
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    
    # 개수 표시
    with st.expander("이미지 타입별 개수"):
        st.dataframe(type_count.to_frame("개수"), use_container_width=True)

def plot_performance_comparison(df_country, country, metric="eng_rate"):
    """성과 비교 차트 (Boxplot)"""
    country_name = get_country_name(country)
    metric_names = {
        "eng_rate": "참여율 (Engagement Rate)",
        "likes": "좋아요 수",
        "comments": "댓글 수"
    }
    
    fig = px.box(
        df_country,
        x="img_type",
        y=metric,
        labels={"img_type": "이미지 타입", metric: metric_names.get(metric, metric)},
        title=f"[{country_name}] 이미지 타입별 {metric_names.get(metric, metric)} 분포"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_performance_summary(agg_perf, country, metric="eng_mean"):
    """성과 요약 차트 (Bar)"""
    country_name = get_country_name(country)
    metric_names = {
        "eng_mean": "평균 참여율",
        "eng_median": "중앙값 참여율"
    }
    
    fig = px.bar(
        agg_perf,
        x="img_type",
        y=metric,
        labels={"img_type": "이미지 타입", metric: metric_names.get(metric, metric)},
        title=f"[{country_name}] 이미지 타입별 {metric_names.get(metric, metric)}"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_top_percentile_probability(prob_df, country, percentile=10):
    """상위 N% 확률 차트"""
    country_name = get_country_name(country)
    
    fig = px.bar(
        prob_df,
        x="img_type",
        y=f"p_top{percentile}",
        labels={"img_type": "이미지 타입", f"p_top{percentile}": f"Top {percentile}% 확률"},
        title=f"[{country_name}] 이미지 타입별 Top {percentile}% 성과 확률"
    )
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

def plot_top_percentile_concentration(concentration_df, country, percentile=10):
    """상위 N% 내 타입 집중도 차트"""
    country_name = get_country_name(country)
    
    fig = px.bar(
        concentration_df,
        x="img_type",
        y=f"share_in_top{percentile}",
        labels={"img_type": "이미지 타입", f"share_in_top{percentile}": f"Top {percentile}% 내 비율"},
        title=f"[{country_name}] Top {percentile}% 성과 내 이미지 타입 구성비"
    )
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

def plot_stability_metrics(stability_df, country, metric="eng_std"):
    """안정성 지표 차트"""
    country_name = get_country_name(country)
    metric_names = {
        "eng_std": "표준편차 (높을수록 변동성 큼)",
        "eng_iqr": "IQR (높을수록 퍼짐 정도 큼)",
        "eng_cv": "변동계수"
    }
    
    fig = px.bar(
        stability_df,
        x="img_type",
        y=metric,
        labels={"img_type": "이미지 타입", metric: metric_names.get(metric, metric)},
        title=f"[{country_name}] 이미지 타입별 {metric_names.get(metric, metric)}"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_usage_vs_performance(merged_df, country, metric="eng_mean"):
    """활용도 vs 성과 산점도"""
    country_name = get_country_name(country)
    metric_names = {
        "eng_mean": "평균 참여율",
        "p_top10": "Top 10% 확률"
    }
    
    fig = px.scatter(
        merged_df,
        x="usage_share",
        y=metric,
        text="img_type",
        labels={"usage_share": "활용 비율", metric: metric_names.get(metric, metric)},
        title=f"[{country_name}] 활용도 vs {metric_names.get(metric, metric)}"
    )
    fig.update_traces(textposition="top center")
    if metric == "p_top10":
        fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

def plot_likes_vs_comments(df_country, country):
    """좋아요 vs 댓글 산점도"""
    country_name = get_country_name(country)
    
    fig = px.scatter(
        df_country,
        x="likes",
        y="comments",
        color="img_type",
        labels={"likes": "좋아요 수", "comments": "댓글 수"},
        title=f"[{country_name}] 좋아요 vs 댓글 (이미지 타입별)"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_comment_ratio(df_country, country):
    """댓글 비율 분포 차트"""
    country_name = get_country_name(country)
    
    fig = px.box(
        df_country,
        x="img_type",
        y="comment_ratio",
        labels={"img_type": "이미지 타입", "comment_ratio": "댓글 비율"},
        title=f"[{country_name}] 이미지 타입별 댓글 비율 분포"
    )
    st.plotly_chart(fig, use_container_width=True)
