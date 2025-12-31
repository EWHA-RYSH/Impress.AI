# ======================================================
# Layout Components — 헤더, 섹션 타이틀
# ======================================================

import streamlit as st

def render_header():
    """앱 헤더 렌더링"""
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 30px;">
            <h1 style="font-size:48px; font-weight:800;">
                Impress<span style="color:#3b82f6;">.AI</span>
            </h1>
            <p style="font-size:18px; color:#6b7280;">
                Image-based Content Performance Insight
            </p>
        </div>
        <hr style="border:none; height:1px; background-color:#e5e7eb; margin-bottom:30px;">
        """,
        unsafe_allow_html=True
    )

