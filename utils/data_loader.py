# ======================================================
# Data Loader — 엑셀 로드, 전처리
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import os

# 데이터 파일 경로
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

@st.cache_data
def load_reference_df():
    """ECDF 계산을 위한 참조 데이터 로드"""
    file_path = os.path.join(DATA_DIR, "agent6_final_reg_db.xlsx")
    df = pd.read_excel(file_path)
    df["log_eng"] = np.log1p(df["eng_rate"])
    return df

@st.cache_data
def load_meta_df():
    """메타데이터 로드"""
    file_path = os.path.join(DATA_DIR, "agent6_final_db.xlsx")
    return pd.read_excel(file_path)

def get_countries(df_meta):
    """국가 목록 반환"""
    return sorted(df_meta["country"].unique())

