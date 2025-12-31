# ======================================================
# EDA Metrics — 국가별 집계, 상위 10%/30%, 안정성, ECDF
# ======================================================

import pandas as pd
import numpy as np

def performance_level(ecdf):
    """ECDF 값에 따른 성과 레벨 반환"""
    if ecdf >= 80:
        return "높음", "badge-high"
    elif ecdf >= 50:
        return "보통", "badge-mid"
    else:
        return "낮음", "badge-low"

def get_country_ecdf_percentile(df_ref, country, pred_logeng):
    """국가별 ECDF percentile 계산"""
    ref = df_ref[df_ref["country"] == country]["log_eng"].dropna().values
    if len(ref) == 0:
        return 50.0
    return float((ref < pred_logeng).mean() * 100.0)

def preprocess_country_data(df, country):
    """국가별 데이터 전처리"""
    df_country = df[df["country"] == country].copy()
    
    # img_type 정리
    df_country = df_country.dropna(subset=["img_type"])
    df_country["img_type"] = pd.to_numeric(df_country["img_type"], errors="coerce").astype("Int64")
    
    # 숫자형 변환
    df_country["likes"] = pd.to_numeric(df_country["likes"], errors="coerce").fillna(0)
    df_country["comments"] = pd.to_numeric(df_country["comments"], errors="coerce").fillna(0)
    df_country["followers"] = pd.to_numeric(df_country["followers"], errors="coerce").replace(0, np.nan)
    
    # 파생 지표
    if "eng_rate" not in df_country.columns or df_country["eng_rate"].isna().all():
        df_country["eng_rate"] = (df_country["likes"] + df_country["comments"]) / df_country["followers"]
    
    df_country["comment_ratio"] = df_country["comments"] / (df_country["likes"] + 1)
    
    return df_country

def get_image_type_distribution(df_country):
    """이미지 타입별 분포 (개수, 비율)"""
    type_count = df_country["img_type"].value_counts().sort_index()
    type_ratio = df_country["img_type"].value_counts(normalize=True).sort_index()
    return type_count, type_ratio

def get_performance_summary(df_country):
    """이미지 타입별 성과 요약 (평균, 중앙값)"""
    agg = df_country.groupby("img_type").agg(
        n=("eng_rate", "size"),
        likes_mean=("likes", "mean"),
        likes_median=("likes", "median"),
        comments_mean=("comments", "mean"),
        comments_median=("comments", "median"),
        eng_mean=("eng_rate", "mean"),
        eng_median=("eng_rate", "median"),
    ).reset_index().sort_values("img_type")
    return agg

def get_top_percentile_metrics(df_country, percentile=10):
    """상위 N% 확률 및 집중도 계산"""
    threshold = df_country["eng_rate"].quantile(1 - percentile/100)
    df_country[f"is_top{percentile}"] = df_country["eng_rate"] >= threshold
    
    # 타입별 Top N% 확률
    prob = df_country.groupby("img_type")[f"is_top{percentile}"].mean().reset_index()
    prob.columns = ["img_type", f"p_top{percentile}"]
    
    # Top N% 내 타입 구성비
    df_top = df_country[df_country[f"is_top{percentile}"]].copy()
    if len(df_top) > 0:
        concentration = df_top["img_type"].value_counts(normalize=True).sort_index().reset_index()
        concentration.columns = ["img_type", f"share_in_top{percentile}"]
    else:
        concentration = pd.DataFrame(columns=["img_type", f"share_in_top{percentile}"])
    
    return prob, concentration, threshold

def get_stability_metrics(df_country):
    """안정성/변동성 지표 계산 (STD, IQR, CV)"""
    def iqr(x):
        return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    
    stability = df_country.groupby("img_type").agg(
        n=("eng_rate", "size"),
        eng_mean=("eng_rate", "mean"),
        eng_std=("eng_rate", "std"),
        eng_iqr=("eng_rate", iqr),
    ).reset_index()
    
    stability["eng_cv"] = stability["eng_std"] / (stability["eng_mean"].replace(0, np.nan))
    return stability.sort_values("img_type")

def get_usage_vs_performance(df_country, top_percentile=10):
    """활용도 vs 성과 분석 (과소/과대 활용 후보 도출)"""
    # 활용도
    usage = df_country["img_type"].value_counts(normalize=True).sort_index().reset_index()
    usage.columns = ["img_type", "usage_share"]
    
    # 평균 성과
    perf = df_country.groupby("img_type")["eng_rate"].mean().reset_index()
    perf.columns = ["img_type", "eng_mean"]
    
    # Top N% 확률
    prob, _, _ = get_top_percentile_metrics(df_country, top_percentile)
    
    # 병합
    merged = usage.merge(perf, on="img_type", how="left")
    merged = merged.merge(prob, on="img_type", how="left")
    
    # Z-score 계산
    merged["usage_share_z"] = (
        (merged["usage_share"] - merged["usage_share"].mean())
        / (merged["usage_share"].std(ddof=0) + 1e-12)
    )
    merged["eng_mean_z"] = (
        (merged["eng_mean"] - merged["eng_mean"].mean())
        / (merged["eng_mean"].std(ddof=0) + 1e-12)
    )
    merged["gap_perf_minus_usage"] = merged["eng_mean_z"] - merged["usage_share_z"]
    
    # 과소 활용 (확대 후보): gap이 큰 순
    underused = merged.nlargest(3, "gap_perf_minus_usage")[
        ["img_type", "usage_share", "eng_mean", f"p_top{top_percentile}", "gap_perf_minus_usage"]
    ]
    
    # 과대 활용 (축소/개선 후보): gap이 작은 순
    overused = merged.nsmallest(3, "gap_perf_minus_usage")[
        ["img_type", "usage_share", "eng_mean", f"p_top{top_percentile}", "gap_perf_minus_usage"]
    ]
    
    return merged, underused, overused

def get_response_characteristics(df_country):
    """반응 성격 분석 (comment ratio, likes vs comments)"""
    comp = df_country.groupby("img_type").agg(
        n=("comment_ratio", "size"),
        comment_ratio_mean=("comment_ratio", "mean"),
        comment_ratio_median=("comment_ratio", "median"),
        comments_mean=("comments", "mean"),
        likes_mean=("likes", "mean"),
    ).reset_index().sort_values("img_type")
    return comp

