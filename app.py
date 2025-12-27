import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="Global Instagram Content Insight Tool",
    layout="wide"
)

st.title("🌍 Global Instagram Content Insight Tool")
st.caption("국가별 인스타그램 콘텐츠 활용도 & 반응 분석 툴")

# -----------------------------
# 데이터 로드
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("agent6_final_db.xlsx")
    return df

df = load_data()

# 안전 체크
required_cols = ["country", "img_type", "eng_rate", "eng_rank_country_type"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"❌ 필수 컬럼 누락: {col}")
        st.stop()

# -----------------------------
# 사이드바
# -----------------------------
st.sidebar.header("🔧 필터 설정")

countries = sorted(df["country"].unique())
selected_country = st.sidebar.selectbox(
    "국가 선택",
    options=["ALL"] + countries
)

if selected_country != "ALL":
    df_view = df[df["country"] == selected_country]
else:
    df_view = df.copy()

# -----------------------------
# 탭 구성
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 활용도 모니터링",
    "🔥 반응 & 성과 분석",
    "🤖 CV 기반 콘텐츠 분류 (데모)"
])

# ======================================================
# TAB 1. 활용도 모니터링
# ======================================================
with tab1:
    st.subheader("📊 이미지 유형 활용도")

    usage = (
        df_view
        .groupby("img_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(usage, use_container_width=True)

    with col2:
        fig, ax = plt.subplots()
        sns.barplot(
            data=usage,
            x="img_type",
            y="count",
            ax=ax
        )
        ax.set_title("Image Type Usage Count")
        ax.set_xlabel("Image Type")
        ax.set_ylabel("Number of Images")
        st.pyplot(fig)

    st.markdown("""
    **해석 포인트**
    - 많이 쓰이는 유형 ≠ 반응이 좋은 유형
    - 국가별 마케팅 전략의 관성 확인 가능
    """)

# ======================================================
# TAB 2. 반응 & 성과 분석
# ======================================================
with tab2:
    st.subheader("🔥 이미지 유형별 반응 성과")

    col1, col2 = st.columns(2)

    # (1) Engagement Rate 분포
    with col1:
        st.markdown("**Engagement Rate 분포**")
        fig, ax = plt.subplots()
        sns.boxplot(
            data=df_view,
            x="img_type",
            y="eng_rate",
            ax=ax
        )
        ax.set_yscale("log")
        ax.set_title("Engagement Rate Distribution (log scale)")
        st.pyplot(fig)

    # (2) 이미지 유형 내 상대 순위
    with col2:
        st.markdown("**이미지 유형 내 상대 순위 (낮을수록 상위)**")
        fig, ax = plt.subplots()
        sns.boxplot(
            data=df_view,
            x="img_type",
            y="eng_rank_country_type",
            ax=ax
        )
        ax.set_title("Relative Rank within Image Type")
        st.pyplot(fig)

    st.markdown("""
    **해석 포인트**
    - 같은 이미지 유형 안에서 누가 상위 성과를 내는가?
    - 국가별 콘텐츠 ‘성공 공식’의 힌트
    """)

# ======================================================
# TAB 3. CV 기반 콘텐츠 분류 (데모)
# ======================================================
with tab3:
    st.subheader("🤖 CV 기반 콘텐츠 분류 (데모 개념)")

    st.markdown("""
    이 탭은 **컴퓨터 비전 모델이 들어갈 자리**입니다.

    ### 현재 단계
    - 이미지 → `img_type` 분류는 **수작업 / 규칙 기반**
    - 분석 전체 기준을 통일하기 위한 목적

    ### 다음 단계 (예선용 충분)
    1. CNN 분류 모델 (6-class)
    2. 새 이미지 업로드
    3. 이미지 유형 예측
    4. 국가별 평균 성과 기반 예상 점수 출력
    """)

    uploaded_file = st.file_uploader(
        "이미지 업로드 (데모)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        st.image(uploaded_file, caption="업로드한 이미지", width=300)
        st.success("👉 (예시) 분류 결과: **제품 + 모델 유형 (Type 4)**")
        st.info("👉 추천 국가: JP, TH")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("AmorePacific AI Challenge | Global Content Insight Tool")
