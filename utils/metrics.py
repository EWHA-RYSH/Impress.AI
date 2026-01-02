# ======================================================
# Metrics — KPI 계산 함수 (캐시된 집계)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from utils.eda_metrics import (
    preprocess_country_data,
    get_image_type_distribution,
    get_performance_summary,
    get_top_percentile_metrics,
    get_stability_metrics,
    get_usage_vs_performance
)

@st.cache_data
def compute_usage_kpis(df_country):
    """Usage Monitor 페이지용 KPI 계산"""
    type_count, type_ratio = get_image_type_distribution(df_country)
    
    # Most Used Type
    most_used_type = type_ratio.idxmax()
    most_used_pct = type_ratio.max() * 100
    
    # Least Used Type
    least_used_type = type_ratio.idxmin()
    least_used_pct = type_ratio.min() * 100
    
    # Engagement Leader
    perf_summary = get_performance_summary(df_country)
    if len(perf_summary) > 0:
        engagement_leader = perf_summary.loc[perf_summary["eng_mean"].idxmax(), "img_type"]
        engagement_value = perf_summary.loc[perf_summary["eng_mean"].idxmax(), "eng_mean"]
    else:
        engagement_leader = None
        engagement_value = 0
    
    return {
        "most_used": {"type": most_used_type, "pct": most_used_pct},
        "least_used": {"type": least_used_type, "pct": least_used_pct},
        "engagement_leader": {"type": engagement_leader, "value": engagement_value}
    }

@st.cache_data
def compute_performance_kpis(df_country):
    """Performance 페이지용 KPI 계산"""
    perf_summary = get_performance_summary(df_country)
    usage_vs_perf, underused, overused = get_usage_vs_performance(df_country, 10)
    stability = get_stability_metrics(df_country)
    
    # Best Engagement Type
    if len(perf_summary) > 0:
        best_type = perf_summary.loc[perf_summary["eng_mean"].idxmax(), "img_type"]
        best_value = perf_summary.loc[perf_summary["eng_mean"].idxmax(), "eng_mean"]
    else:
        best_type = None
        best_value = 0
    
    # Underused Opportunity
    if len(underused) > 0:
        underused_type = int(underused.iloc[0]["img_type"])
        underused_eng = underused.iloc[0]["eng_mean"]
        underused_usage = underused.iloc[0]["usage_share"] * 100
    else:
        underused_type = None
        underused_eng = 0
        underused_usage = 0
    
    # Stability (평균 IQR 기준)
    if len(stability) > 0:
        avg_iqr = stability["eng_iqr"].mean()
        median_iqr = stability["eng_iqr"].median()
        is_stable = avg_iqr < median_iqr
        stability_label = "Stable" if is_stable else "Volatile"
    else:
        stability_label = "N/A"
    
    return {
        "best_engagement": {"type": best_type, "value": best_value},
        "underused_opportunity": {
            "type": underused_type,
            "engagement": underused_eng,
            "usage": underused_usage
        },
        "stability": {"label": stability_label}
    }

def format_engagement_rate(value):
    """참여율 포맷팅"""
    if pd.isna(value) or value == 0:
        return "N/A"
    return f"{value:.4f}"

def format_percentage(value):
    """퍼센트 포맷팅"""
    return f"{value:.1f}%"


