# ======================================================
# Impress.AI — 앱 엔트리 (라우팅 + 레이아웃만)
# ======================================================

import streamlit as st

from components.layout import render_header
from components.style import inject_style
from utils.data_loader import load_reference_df, load_meta_df, get_countries
from tabs.tab1_usage import render as render_tab1
from tabs.tab2_performance import render as render_tab2
from tabs.tab3_predict import render as render_tab3

# ======================================================
# Page Config
# ======================================================
st.set_page_config(
    page_title="Impress.AI",
    page_icon="📸",
    layout="wide"
)

# ======================================================
# Header
# ======================================================
render_header()

# ======================================================
# Load Data
# ======================================================
df_ref = load_reference_df()
df_meta = load_meta_df()
countries = get_countries(df_meta)

# ======================================================
# Sidebar
# ======================================================
st.sidebar.header("🔧 Filters")
selected_country = st.sidebar.selectbox("Select Country", countries)
st.session_state.selected_country = selected_country
st.sidebar.caption(
    f"📊 Records: {len(df_meta[df_meta['country']==selected_country])}"
)

# ======================================================
# Tabs
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "📊 콘텐츠 활용 모니터링",
    "🔥 콘텐츠 성과 분석 & 패턴 도출",
    "🤖 AI 콘텐츠 성과 예측"
])

# ======================================================
# Tab Rendering
# ======================================================
with tab1:
    render_tab1()

with tab2:
    render_tab2()

with tab3:
    render_tab3(df_ref)
