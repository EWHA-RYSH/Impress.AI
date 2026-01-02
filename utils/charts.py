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
    "#9CA3AF",  # Gray (기본)
    "#6B7280",  # Gray (다크)
    "#D1D5DB",  # Gray (라이트)
    "#E5E7EB",  # Gray (더 라이트)
    "#F3F4F6"   # Gray (가장 라이트)
]

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
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
        title=dict(
            x=0.5,
            xanchor="center",
            font=dict(size=15, color="#111827", family="Arita-Dotum-Medium, sans-serif")
        ),
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
            zeroline=False
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
    
    # 강조할 타입이 있으면 컬러 지정
    colors = []
    for idx, img_type in enumerate(type_ratio.index):
        if highlight_type and str(highlight_type) == str(img_type):
            colors.append(BRAND_COLORS["primary"])
        else:
            colors.append(CHART_PALETTE[0])
    
    fig = px.bar(
        df_plot,
        x="Image Type",
        y="Usage Share",
        labels={"Image Type": "이미지 타입", "Usage Share": "활용 비율"},
        title="이미지 타입별 활용 분포"
    )
    
    fig.update_traces(marker_color=colors)
    fig = apply_chart_style(fig, highlight_type)
    fig.update_layout(
        yaxis_tickformat=".0%",
        xaxis=dict(
            tickangle=0,
            title=dict(
                font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                standoff=12
            )
        ),
        yaxis=dict(
            title=dict(
                font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12)
            )
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def plot_engagement_distribution(df_country, country, highlight_type=None):
    """참여율 분포 (Scatter Plot with Log Scale)"""
    import numpy as np
    from components.layout import get_type_name
    
    country_name = get_country_name(country)
    
    # 데이터 복사
    df_plot = df_country.copy()
    
    # 타입명으로 변환하여 x축 레이블 생성
    df_plot["type_label"] = df_plot["img_type"].apply(
        lambda x: f"{x}. {get_type_name(x)}"
    )
    
    # 참여율에 작은 상수를 더해서 로그 변환 가능하게 함 (0 값 처리)
    # 로그 스케일에서 0에 가까운 값들이 잘 보이도록
    epsilon = 1e-6
    df_plot["eng_rate_adj"] = df_plot["eng_rate"] + epsilon
    
    # 색상 및 크기 설정
    if highlight_type:
        colors = [
            BRAND_COLORS["primary"] if str(img_type) == str(highlight_type) else CHART_PALETTE[0]
            for img_type in df_plot["img_type"]
        ]
        sizes = [
            10 if str(img_type) == str(highlight_type) else 6
            for img_type in df_plot["img_type"]
        ]
    else:
        colors = [CHART_PALETTE[0]] * len(df_plot)
        sizes = [6] * len(df_plot)
    
    # scatter plot 생성
    fig = px.scatter(
        df_plot,
        x="type_label",
        y="eng_rate_adj",
        labels={"type_label": "이미지 타입", "eng_rate_adj": "참여율"},
        title="이미지 타입별 참여율 분포"
    )
    
    # 색상 및 크기 적용
    fig.update_traces(
        marker=dict(
            color=colors,
            size=sizes,
            opacity=0.7
        )
    )
    
    # 공통 스타일 적용
    fig = apply_chart_style(fig, highlight_type)
    
    # 로그 스케일 적용 (apply_chart_style 이후에 적용하여 덮어쓰기)
    # 더 읽기 쉬운 형식으로 표시
    fig.update_layout(
        margin=dict(b=50),  # 하단 마진
        xaxis=dict(
            tickangle=0,
            title=dict(
                font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                standoff=20
            )
        ),
        yaxis=dict(
            type="log",
            showgrid=True,
            gridcolor="#F3F4F6",
            gridwidth=1,
            zeroline=False,
            title=dict(
                text="참여율",
                font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12)
            ),
            tickformat=".4f"
        )
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
        
        colors = []
        for idx, img_type in enumerate(usage_df.index):
            if highlight_type and str(highlight_type) == str(img_type):
                colors.append(BRAND_COLORS["primary"])
            else:
                colors.append(CHART_PALETTE[0])
        
        fig1 = px.bar(
            df_usage,
            x="Image Type",
            y="Usage Share",
            labels={"Image Type": "이미지 타입", "Usage Share": "활용 비율"},
            title="활용 비율"
        )
        fig1.update_traces(marker_color=colors)
        fig1 = apply_chart_style(fig1)
        fig1.update_layout(
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
        
        colors = []
        for img_type in df_eng["img_type"]:
            if highlight_type and str(highlight_type) == str(img_type):
                colors.append(BRAND_COLORS["primary"])
            else:
                colors.append(CHART_PALETTE[0])
        
        fig2 = px.bar(
            df_eng,
            x="type_label",
            y="eng_mean",
            labels={"type_label": "이미지 타입", "eng_mean": "평균 참여율"},
            title="평균 참여율"
        )
        fig2.update_traces(marker_color=colors)
        fig2 = apply_chart_style(fig2)
        fig2.update_layout(
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
