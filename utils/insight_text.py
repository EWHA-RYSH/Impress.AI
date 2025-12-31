# ======================================================
# Insight Text — 자동 인사이트 문구 생성
# ======================================================

import pandas as pd

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

# 이미지 타입 설명
TYPE_DESC = {
    1: "여러 제품을 함께 보여주는 제품 단체샷",
    2: "한 제품을 단독으로 강조한 제품 단독샷",
    3: "제품 제형/텍스처를 중심으로 한 제품 질감샷",
    4: "모델과 제품을 함께 배치한 이미지",
    5: "제품 없이 모델 중심으로 연출된 이미지",
    6: "여러 인물과 제품을 함께 보여주는 이미지"
}

def get_country_name(code):
    """국가 코드를 한글 이름으로 변환"""
    return COUNTRY_NAMES.get(code, code)

def generate_usage_insights(type_count, type_ratio, country):
    """활용도 인사이트 생성"""
    country_name = get_country_name(country)
    top2 = type_count.head(2)
    
    insights = []
    insights.append(f"### [{country_name}] 콘텐츠 활용 현황")
    insights.append(f"\n**가장 많이 사용된 이미지 타입 TOP 2:**")
    
    for idx, (img_type, count) in enumerate(top2.items(), 1):
        ratio = type_ratio[img_type] * 100
        type_name = TYPE_DESC.get(int(img_type), f"Type {img_type}")
        insights.append(f"{idx}. **Type {img_type}** ({type_name}): {count}개 ({ratio:.1f}%)")
    
    return "\n".join(insights)

def generate_performance_insights(agg_perf, country):
    """성과 인사이트 생성"""
    country_name = get_country_name(country)
    best_type = agg_perf.loc[agg_perf["eng_mean"].idxmax(), "img_type"]
    best_mean = agg_perf.loc[agg_perf["eng_mean"].idxmax(), "eng_mean"]
    
    insights = []
    insights.append(f"### [{country_name}] 성과 분석")
    insights.append(f"\n**평균 참여율이 가장 높은 이미지 타입:**")
    type_name = TYPE_DESC.get(int(best_type), f"Type {best_type}")
    insights.append(f"- **Type {best_type}** ({type_name}): 평균 참여율 {best_mean:.6f}")
    
    return "\n".join(insights)

def generate_top_percentile_insights(prob_df, concentration_df, country, percentile=10):
    """상위 N% 인사이트 생성"""
    country_name = get_country_name(country)
    
    insights = []
    insights.append(f"### [{country_name}] 고성과 콘텐츠 분석")
    
    # Top N% 확률이 가장 높은 타입
    if len(prob_df) > 0:
        best_prob_type = prob_df.loc[prob_df[f"p_top{percentile}"].idxmax(), "img_type"]
        best_prob = prob_df.loc[prob_df[f"p_top{percentile}"].idxmax(), f"p_top{percentile}"]
        type_name = TYPE_DESC.get(int(best_prob_type), f"Type {best_prob_type}")
        insights.append(f"\n**Top {percentile}% 성과 달성 확률이 가장 높은 타입:**")
        insights.append(f"- **Type {best_prob_type}** ({type_name}): {best_prob*100:.1f}%")
    
    # Top N% 내에서 가장 많이 등장하는 타입
    if len(concentration_df) > 0:
        best_conc_type = concentration_df.loc[concentration_df[f"share_in_top{percentile}"].idxmax(), "img_type"]
        best_conc = concentration_df.loc[concentration_df[f"share_in_top{percentile}"].idxmax(), f"share_in_top{percentile}"]
        type_name = TYPE_DESC.get(int(best_conc_type), f"Type {best_conc_type}")
        insights.append(f"\n**Top {percentile}% 성과 내에서 가장 많이 등장하는 타입:**")
        insights.append(f"- **Type {best_conc_type}** ({type_name}): {best_conc*100:.1f}%")
    
    return "\n".join(insights)

def generate_stability_insights(stability_df, country):
    """안정성 인사이트 생성"""
    country_name = get_country_name(country)
    most_stable = stability_df.loc[stability_df["eng_std"].idxmin(), "img_type"]
    most_stable_std = stability_df.loc[stability_df["eng_std"].idxmin(), "eng_std"]
    
    insights = []
    insights.append(f"### [{country_name}] 성과 안정성 분석")
    insights.append(f"\n**가장 안정적인 이미지 타입 (표준편차 최소):**")
    type_name = TYPE_DESC.get(int(most_stable), f"Type {most_stable}")
    insights.append(f"- **Type {most_stable}** ({type_name}): 표준편차 {most_stable_std:.6f}")
    
    return "\n".join(insights)

def generate_strategy_insights(underused_df, overused_df, country):
    """전략 인사이트 생성 (과소/과대 활용)"""
    country_name = get_country_name(country)
    
    insights = []
    insights.append(f"### [{country_name}] 전략적 개선 포인트")
    
    if len(underused_df) > 0:
        insights.append(f"\n**과소 활용 (확대 후보) TOP 3:**")
        for idx, row in underused_df.iterrows():
            type_name = TYPE_DESC.get(int(row["img_type"]), f"Type {row['img_type']}")
            insights.append(
                f"{int(row.name) + 1}. **Type {int(row['img_type'])}** ({type_name}): "
                f"활용도 {row['usage_share']*100:.1f}%, 평균 성과 {row['eng_mean']:.6f}"
            )
    
    if len(overused_df) > 0:
        insights.append(f"\n**과대 활용 (축소/개선 후보) TOP 3:**")
        for idx, row in overused_df.iterrows():
            type_name = TYPE_DESC.get(int(row["img_type"]), f"Type {row['img_type']}")
            insights.append(
                f"{int(row.name) + 1}. **Type {int(row['img_type'])}** ({type_name}): "
                f"활용도 {row['usage_share']*100:.1f}%, 평균 성과 {row['eng_mean']:.6f}"
            )
    
    return "\n".join(insights)

def generate_summary_insights(df_country, country, all_metrics):
    """종합 인사이트 요약"""
    country_name = get_country_name(country)
    
    insights = []
    insights.append(f"## [{country_name}] 최종 인사이트 요약")
    insights.append("\n" + "="*60)
    
    # 주요 발견사항
    insights.append("\n### 주요 발견사항")
    
    # 가장 많이 사용된 타입
    type_count, type_ratio = all_metrics.get("type_distribution", (None, None))
    if type_count is not None and len(type_count) > 0:
        most_used = type_count.index[0]
        most_used_name = TYPE_DESC.get(int(most_used), f"Type {most_used}")
        insights.append(f"- 가장 많이 사용된 타입: **Type {most_used}** ({most_used_name})")
    
    # 평균 성과가 가장 높은 타입
    agg_perf = all_metrics.get("performance_summary")
    if agg_perf is not None and len(agg_perf) > 0:
        best_perf = agg_perf.loc[agg_perf["eng_mean"].idxmax(), "img_type"]
        best_perf_name = TYPE_DESC.get(int(best_perf), f"Type {best_perf}")
        insights.append(f"- 평균 성과가 가장 높은 타입: **Type {best_perf}** ({best_perf_name})")
    
    # Top 10% 확률이 가장 높은 타입
    prob_df = all_metrics.get("top10_prob")
    if prob_df is not None and len(prob_df) > 0:
        best_prob = prob_df.loc[prob_df["p_top10"].idxmax(), "img_type"]
        best_prob_name = TYPE_DESC.get(int(best_prob), f"Type {best_prob}")
        insights.append(f"- Top 10% 달성 확률이 가장 높은 타입: **Type {best_prob}** ({best_prob_name})")
    
    return "\n".join(insights)
