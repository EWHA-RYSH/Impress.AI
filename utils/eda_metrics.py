# ======================================================
# EDA Metrics — 국가별 집계, 상위 10%/30%, 안정성, ECDF
# ======================================================

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

