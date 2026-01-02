# ======================================================
# Charts — 공통 차트 함수 (Enterprise B2B SaaS)
# ======================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 브랜드 컬러 팔레트
BRAND_COLORS = {
    "primary": "#1F5795",      # Amore Blue
    "deep": "#001C58",          # Pacific Blue
    "gray": "#7D7D7D",          # Gray
    "black": "#000000",         # Black
    "white": "#FFFFFF",         # White
    "bg": "#F7F8FA"             # Background
}

# 차트용 컬러 팔레트 
CHART_PALETTE = [
    "#DADDE3",  # 연한 회색 (기본 막대)
    "#E3E6EC",  # 더 연한 회색
    "#9CA3AF",  # Gray (기본)
    "#6B7280",  # Gray (다크)
    "#D1D5DB",  # Gray (라이트)
    "#E5E7EB",  # Gray (더 라이트)
    "#F3F4F6"   # Gray (가장 라이트)
]

# 차트 강조 색상
HIGHLIGHT_COLOR = "#4C6FBF"  # 채도 낮춘 블루 (강조 막대)
DEFAULT_BAR_COLOR = "#E1E4EA"  # 막대 1개 기본 색상
LIGHT_BLUE_HIGHLIGHT = "#B9CBE3"  # 아주 연한 블루 (Top 1 강조용)

# 막대 2개짜리 그래프 색상
MEDIAN_COLOR = "#E5E7EB"  # 중앙값 색상
MEAN_COLOR = "#9CA3AF"  # 평균 색상

TEMPLATE = "plotly_white"

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

def apply_chart_style(fig, highlight_type=None):
    """차트 공통 스타일 적용"""
    fig.update_layout(
        template=TEMPLATE,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12, color="#374151"),
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        title=None,
        xaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            gridwidth=1,
            zeroline=False,
            title=dict(
                font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                standoff=8
            )
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            gridwidth=1,
            zeroline=False,
            title=None
        )
    )
    # 모드바 숨김
    fig.update_layout(modebar_remove=["zoom", "pan", "select", "lasso", "autoScale", "reset"])
    return fig

def plot_usage_distribution(type_ratio, country, highlight_type=None):
    """이미지 타입별 활용 분포 (Bar Chart)"""
    from components.layout import get_type_name
    
    country_name = get_country_name(country)
    
    # 타입명으로 변환
    type_labels = []
    for img_type in type_ratio.index:
        type_name = get_type_name(img_type)
        type_labels.append(f"{img_type}. {type_name}")
    
    # 데이터 준비
    df_plot = pd.DataFrame({
        "Image Type": type_labels,
        "Usage Share": type_ratio.values
    })
    
    # 최고값 막대는 연한 하늘색으로 강조
    max_idx = type_ratio.idxmax()
    colors = []
    text_values = []
    for img_type in type_ratio.index:
        if highlight_type is not None and img_type == highlight_type:
            colors.append(LIGHT_BLUE_HIGHLIGHT)  # 최고값은 연한 하늘색
        else:
            colors.append(DEFAULT_BAR_COLOR)  # 나머지는 #E1E4EA
        text_values.append(f"{type_ratio[img_type]*100:.1f}%")
    
    fig = px.bar(
        df_plot,
        x="Image Type",
        y="Usage Share",
        labels={"Image Type": "이미지 타입", "Usage Share": ""},
        title="이미지 타입별 활용 분포",
        text=text_values
    )
    
    fig.update_traces(
        marker_color=colors,
        width=0.6,  # 막대 폭 줄이기 (기본값보다 약 20-30% 얇게)
        textposition="outside",
        textfont=dict(size=11, color="#6B7280", family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif")
    )
    fig = apply_chart_style(fig, highlight_type)
    fig.update_layout(
        bargap=0.4,  # 막대 간격 조정 (0.35 ~ 0.45 범위)
        showlegend=False,
        yaxis_tickformat=".0%",
        yaxis=dict(title=None),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(
            tickangle=0,
            title=dict(
                font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                standoff=12
            )
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def plot_engagement_distribution(df_country, country, highlight_type=None):
    """참여율 분포 (Bar Chart - 평균 참여율)"""
    from utils.eda_metrics import get_performance_summary
    from components.layout import get_type_name
    
    country_name = get_country_name(country)
    
    # 성과 요약 데이터 가져오기
    perf_summary = get_performance_summary(df_country)
    
    if len(perf_summary) == 0:
        st.info("참여율 데이터가 없습니다.")
        return
    
    # 최고값 막대는 연한 하늘색으로 강조
    max_idx = perf_summary["eng_mean"].idxmax()
    colors = []
    text_values = []
    for idx, row in perf_summary.iterrows():
        if highlight_type is not None and row["img_type"] == highlight_type:
            colors.append(LIGHT_BLUE_HIGHLIGHT)  # 최고값은 연한 하늘색
        else:
            colors.append(DEFAULT_BAR_COLOR)  # 나머지는 #E1E4EA
        # 값 라벨 추가 (참여율은 소수점 표시)
        text_values.append(f"{row['eng_mean']:.4f}")
    
    fig = px.bar(
        perf_summary,
        x="img_type",
        y="eng_mean",
        labels={"img_type": "이미지 타입", "eng_mean": ""},
        title="이미지 타입별 평균 참여율",
        text=text_values
    )
    fig.update_traces(
        marker_color=colors, 
        width=0.6,
        textposition="outside",
        textfont=dict(size=11, color="#6B7280", family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif")
    )
    fig = apply_chart_style(fig)
    fig.update_layout(
        bargap=0.4, 
        showlegend=False, 
        height=400,
        yaxis=dict(title=None),
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def plot_usage_vs_engagement(usage_df, perf_df, country, highlight_type=None):
    """활용도 vs 참여율 (Side-by-side)"""
    from components.layout import get_type_name
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Usage Bar
        type_labels = [str(img_type) for img_type in usage_df.index]
        
        df_usage = pd.DataFrame({
            "Image Type": type_labels,
            "Usage Share": usage_df.values
        })
        
        # 모든 막대를 회색 계열로 통일 (데이터 비교는 중립색)
        colors = [CHART_PALETTE[0]] * len(usage_df)
        
        fig1 = px.bar(
            df_usage,
            x="Image Type",
            y="Usage Share",
            labels={"Image Type": "이미지 타입", "Usage Share": "활용 비율"},
            title="활용 비율"
        )
        fig1.update_traces(
            marker_color=colors,
            width=0.6  # 막대 폭 줄이기
        )
        fig1 = apply_chart_style(fig1)
        fig1.update_layout(
            bargap=0.4,  # 막대 간격 조정
            yaxis_tickformat=".0%",
            xaxis=dict(
                tickangle=0,
                title=dict(
                    font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                    standoff=8
                )
            ),
            yaxis=dict(
                title=dict(
                    font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12)
                )
            )
        )
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
    
    with col2:
        # Engagement Bar (perf_df에서 img_type과 eng_mean 사용)
        df_eng = perf_df[["img_type", "eng_mean"]].copy()
        df_eng["type_label"] = df_eng["img_type"].apply(str)
        
        # 모든 막대를 회색 계열로 통일 (데이터 비교는 중립색)
        colors = [CHART_PALETTE[0]] * len(df_eng)
        
        fig2 = px.bar(
            df_eng,
            x="type_label",
            y="eng_mean",
            labels={"type_label": "이미지 타입", "eng_mean": "평균 참여율"},
            title="평균 참여율"
        )
        fig2.update_traces(
            marker_color=colors,
            width=0.6  # 막대 폭 줄이기
        )
        fig2 = apply_chart_style(fig2)
        fig2.update_layout(
            bargap=0.4,  # 막대 간격 조정
            xaxis=dict(
                tickangle=0,
                title=dict(
                    font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                    standoff=8
                )
            ),
            yaxis=dict(
                title=dict(
                    font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12)
                )
            )
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
