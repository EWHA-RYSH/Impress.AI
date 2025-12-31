# ======================================================
# Style Components — CSS 한 번만 주입
# ======================================================

import streamlit as st

def inject_style():
    """CSS 스타일을 한 번만 주입"""
    st.markdown(
        """
        <style>
        .result-card {
            background: #ffffff;
            padding: 28px;
            border-radius: 20px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 10px 24px rgba(0,0,0,0.06);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", sans-serif;
        }

        .badge-high, .badge-mid, .badge-low {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 14px;
        }

        .badge-high { background: #dcfce7; color: #166534; }
        .badge-mid  { background: #fef9c3; color: #854d0e; }
        .badge-low  { background: #fee2e2; color: #991b1b; }

        .muted { color:#6b7280; }
        .hr { border:none; height:1px; background:#e5e7eb; margin:18px 0; }
        .h1 { font-size: 40px; margin: 8px 0 6px; font-weight: 800; }
        .h2 { font-size: 26px; margin: 0 0 6px; font-weight: 800; }
        .h4 { font-size: 16px; margin: 14px 0 6px; font-weight: 800; }
        .small { color:#6b7280; font-size:13px; line-height:1.45; }
        </style>
        """,
        unsafe_allow_html=True
    )

