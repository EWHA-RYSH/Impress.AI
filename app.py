import streamlit as st

from components.style import apply_custom_style
from utils.data_loader import load_reference_df, load_meta_df, get_countries
from tabs.tab1_usage import render as render_tab1
from tabs.tab2_performance import render as render_tab2
from tabs.tab3_predict import render as render_tab3

st.set_page_config(
    page_title="AP.SIGNAL",
    page_icon="📸",
    layout="wide"
)

apply_custom_style()

df_ref = load_reference_df()
df_meta = load_meta_df()
countries = get_countries(df_meta)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title" style="padding: 20px 0 20px 12px;">
            <div class="sidebar-title-text" style="font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', sans-serif !important; font-size: 28px; font-weight: 700; color: #FFFFFF;">
                AP.SIGNAL
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 구분선 추가
    st.markdown(
        """
        <div style="
            height: 1px;
            background-color: rgba(255, 255, 255, 0.2);
            margin: 0 16px 24px 16px;
        "></div>
        """,
        unsafe_allow_html=True
    )
    
    nav_options = ["활용도 모니터링", "성과 분석", "AI 예측"]
    
    if "nav_selected" not in st.session_state:
        st.session_state.nav_selected = 0
    
    selected_idx = st.radio(
        "Navigation",
        nav_options,
        index=st.session_state.nav_selected,
        label_visibility="collapsed",
        key="sidebar_nav"
    )
    st.session_state.nav_selected = nav_options.index(selected_idx)
    
    nav_map = {
        "활용도 모니터링": "Usage Monitor",
        "성과 분석": "Performance",
        "AI 예측": "AI Prediction"
    }
    selected = nav_map[selected_idx]

if "selected_country" not in st.session_state:
    st.session_state.selected_country = countries[0]

if selected == "Usage Monitor":
    render_tab1()
elif selected == "Performance":
    render_tab2()
elif selected == "AI Prediction":
    render_tab3(df_ref)
