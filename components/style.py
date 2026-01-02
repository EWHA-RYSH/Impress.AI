import streamlit as st
import base64
import os

BRAND_COLORS = {
    "primary": "#1F5795",
    "deep": "#001C58",
    "gray": "#7D7D7D",
    "black": "#000000",
    "white": "#FFFFFF",
    "bg": "#F7F8FA"
}

def get_font_base64(font_path):
    try:
        with open(font_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

def inject_style():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    font_dir = os.path.join(base_dir, "fonts")
    
    fonts = {
        "Arita-Dotum-Medium": "AritaDotumKR-Medium.ttf",
        "Arita-Dotum-Bold": "AritaDotumKR-Bold.ttf",
        "Arita-Sans-Medium": "AritaSansLTN-Medium.ttf",
        "Arita-Sans-Bold": "AritaSansLTN-Bold.ttf",
    }
    
    font_css = ""
    for font_name, filename in fonts.items():
        path = os.path.join(font_dir, filename)
        b64 = get_font_base64(path)
        if b64:
            font_css += f"""
            @font-face {{
                font-family: '{font_name}';
                src: url(data:font/ttf;charset=utf-8;base64,{b64}) format('truetype');
                font-weight: {'bold' if 'Bold' in font_name else 'normal'};
                font-style: normal;
            }}
            """
    
    webfont_css = """
        @font-face {
            font-family: 'Arita-Sans-Medium';
            src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_one@1.0/AritaSans-Medium.woff') format('woff');
            font-weight: normal;
            font-style: normal;
        }
        @font-face {
            font-family: 'Arita-Sans-Bold';
            src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_one@1.0/AritaSans-Bold.woff') format('woff');
            font-weight: bold;
            font-style: normal;
        }
        @font-face {
            font-family: 'Arita-dotum-Medium';
            src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_one@1.0/Arita-dotum-Medium.woff') format('woff');
            font-weight: normal;
            font-style: normal;
        }
        @font-face {
            font-family: 'Arita-Dotum-Bold';
            src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_one@1.0/Arita-dotum-Bold.woff') format('woff');
            font-weight: bold;
            font-style: normal;
        }
    """
    
    st.markdown(
        f"""
        <style>
        {webfont_css}
        {font_css}
        
        * {{
            font-family: 'Arita-Sans-Medium', sans-serif !important;
        }}
        
        html, body, div, span, p, a, li, ul, ol, td, th, label, input, textarea, select, button {{
            font-family: 'Arita-Sans-Medium', sans-serif !important;
        }}
        
        .stApp, .stApp > div, [class*="st"], [class*="css"], [data-testid] {{
            font-family: 'Arita-Sans-Medium', sans-serif !important;
        }}
        
        .stMarkdown, .stMarkdown *, .stText, .stText *, .stCaption, .stCaption * {{
            font-family: 'Arita-Sans-Medium', sans-serif !important;
        }}
        
        /* KPI 카드 전용 폰트 - 클래스 기반 */
        .kpi-card-wrapper,
        .kpi-card-wrapper * {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;
        }}
        
        .kpi-value {{
            font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Bold', 'Arita-Dotum-Bold', 'Malgun Gothic', sans-serif !important;
        }}
        
        /* KPI 카드 전용 폰트 - 전역 스타일보다 우선순위 높게 */
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"],
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"] div,
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"] span {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;
        }}
        
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"] div[style*="font-weight: 700"],
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"] div[style*="font-weight: 600"] {{
            font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Bold', 'Arita-Dotum-Bold', 'Malgun Gothic', sans-serif !important;
        }}
        
        h1, h2, h3, h4, h5, h6, .h1, .h2, .h3, .h4 {{
            font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', sans-serif !important;
            font-weight: 700 !important;
        }}
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
            font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', sans-serif !important;
            font-weight: 700 !important;
        }}
        
        .page-title,
        [data-testid="stMarkdownContainer"] .page-title,
        [data-testid="stMarkdownContainer"] div.page-title {{
            font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', sans-serif !important;
            font-weight: 800 !important;
        }}
        
        .page-description,
        [data-testid="stMarkdownContainer"] .page-description,
        [data-testid="stMarkdownContainer"] div.page-description {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
            font-weight: 400 !important;
        }}
        
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] h5,
        [data-testid="stMarkdownContainer"] h6 {{
            font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', sans-serif !important;
            font-weight: 700 !important;
        }}
        
        strong, b {{
            font-family: 'Arita-Sans-Bold', sans-serif !important;
            font-weight: 700 !important;
        }}
        
        .kpi-card, .kpi-card *, [class*="insight"], [class*="action"] {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"] {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"] * {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #F9FAFB"][style*="border-left: 4px solid"] {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #F9FAFB"][style*="border-left: 4px solid"] * {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        /* KPI 카드 내부 모든 텍스트에 Arita Dotum 적용 - 더 강력한 선택자 */
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"][style*="border-radius: 8px"],
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"][style*="border-radius: 8px"] *,
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"][style*="padding: 20px"] * {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        [data-testid="stSidebar"], [data-testid="stSidebar"] * {{
            font-family: 'Arita-Sans-Medium', sans-serif !important;
        }}
        
        [data-testid="stSidebar"] .sidebar-title,
        [data-testid="stSidebar"] .sidebar-title *,
        [data-testid="stSidebar"] .sidebar-title-text,
        [data-testid="stSidebar"] .sidebar-title-text * {{
            font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', sans-serif !important;
            font-weight: 700 !important;
        }}
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {{
            font-family: 'Arita-Sans-Bold', sans-serif !important;
            font-weight: 700 !important;
        }}
        
        [data-testid="stSidebar"] hr {{
            margin: 20px 0 !important;
        }}
        
        [data-testid="stSidebar"] .stRadio {{
            margin: 0 !important;
            padding-left: 0 !important;
            margin-top: 8px !important;
        }}
        
        [data-testid="stSidebar"] .stRadio > div > label {{
            padding-left: 12px !important;
            padding-right: 12px !important;
        }}
        
        [data-testid="stSidebar"] hr + .stRadio {{
            margin-top: 0 !important;
        }}
        
        [data-testid="stSidebar"] .stRadio + hr {{
            margin-top: 20px !important;
            margin-bottom: 20px !important;
        }}
        
        /* 사이드바 타이틀과 메뉴 사이 여백 */
        [data-testid="stSidebar"] .sidebar-title {{
            margin-bottom: 0 !important;
        }}
        
        [data-baseweb="select"], [data-baseweb="button"], [data-baseweb="input"] {{
            font-family: 'Arita-Sans-Medium', sans-serif !important;
        }}

        .stApp {{
            background-color: {BRAND_COLORS['bg']};
        }}

        div[data-testid="metric-container"] {{
            background-color: {BRAND_COLORS['white']};
            border: 1px solid #E5E7EB;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        
        .kpi-card {{
            background-color: {BRAND_COLORS['white']};
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            margin-bottom: 16px;
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        .kpi-card * {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        .kpi-card.highlight {{
            border-left: 4px solid {BRAND_COLORS['primary']};
        }}
        
        /* KPI 카드 스타일이 적용된 div 내부 모든 요소 - 최대한 강력한 선택자 */
        div[data-testid="stMarkdownContainer"] > div > div[style*="background-color: #FFFFFF"],
        div[data-testid="stMarkdownContainer"] > div > div[style*="background-color: #FFFFFF"] *,
        div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"],
        div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"] *,
        div[style*="background-color: #FFFFFF"][style*="border-radius: 8px"],
        div[style*="background-color: #FFFFFF"][style*="border-radius: 8px"] *,
        div[style*="background-color: #FFFFFF"][style*="padding: 20px"],
        div[style*="background-color: #FFFFFF"][style*="padding: 20px"] * {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;
        }}
        
        /* Bold 텍스트는 Bold 폰트 사용 */
        div[style*="background-color: #FFFFFF"] div[style*="font-weight: 700"],
        div[style*="background-color: #FFFFFF"] div[style*="font-weight: 600"],
        div[style*="background-color: #FFFFFF"] div[style*="font-weight: 700"] *,
        div[style*="background-color: #FFFFFF"] div[style*="font-weight: 600"] * {{
            font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Bold', 'Arita-Dotum-Bold', 'Malgun Gothic', sans-serif !important;
        }}
        
        .stPlotlyChart {{
            background-color: {BRAND_COLORS['white']};
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            overflow: hidden !important;
            max-width: 100% !important;
        }}
        
        .stPlotlyChart > div {{
            overflow: hidden !important;
            max-width: 100% !important;
        }}
        
        .stPlotlyChart > div > div {{
            overflow: hidden !important;
            max-width: 100% !important;
            width: 100% !important;
        }}
        
        div.stDataFrame {{
            background-color: {BRAND_COLORS['white']};
            border: 1px solid #E5E7EB;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            overflow: hidden !important;
            max-width: 100% !important;
        }}
        
        div.stDataFrame > div {{
            overflow-x: auto !important;
            overflow-y: visible !important;
            max-width: 100% !important;
        }}
        
        div.stDataFrame table {{
            max-width: 100% !important;
            width: 100% !important;
        }}
        
        div.stDataFrame table th {{
            text-align: center !important;
            font-weight: 500 !important;
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        div.stDataFrame table td {{
            text-align: right !important;
            font-weight: 400 !important;
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        div.stDataFrame table td:first-child {{
            text-align: center !important;
        }}
        
        div.stDataFrame table td:last-child {{
            background-color: #F9FAFB !important;
        }}
        
        div.stDataFrame table th:last-child {{
            background-color: #F3F4F6 !important;
        }}
        
        .image-type-card img {{
            aspect-ratio: 1;
            object-fit: cover;
            border-radius: 8px;
        }}
        
        div[data-testid="stImage"] {{
            margin: 0 !important;
        }}
        
        div[data-testid="stImage"] img {{
            width: 100% !important;
            height: auto !important;
            object-fit: cover !important;
            border-radius: 8px !important;
            max-height: 200px !important;
        }}
        
        .js-plotly-plot {{
            overflow: hidden !important;
            max-width: 100% !important;
        }}
        
        [data-testid="column"] {{
            overflow: hidden !important;
            max-width: 100% !important;
        }}
        
        [data-testid="column"] > div {{
            overflow: hidden !important;
            max-width: 100% !important;
            width: 100% !important;
        }}
        
        .stMarkdown p, .stMarkdown div {{
            word-break: keep-all;
            overflow-wrap: break-word;
        }}
        
        [data-testid="column"]:last-child .stMarkdown {{
            min-width: fit-content;
        }}
        
        .main .block-container {{
            max-width: 100% !important;
            overflow-x: hidden !important;
        }}
        
        [data-testid="stSidebar"] {{
            background-color: {BRAND_COLORS['primary']} !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        [data-testid="stSidebar"] * {{
            color: #FFFFFF !important;
        }}
        
        [data-testid="stSidebar"] .sidebar-title,
        [data-testid="stSidebar"] .sidebar-title *,
        [data-testid="stSidebar"] .sidebar-title-text,
        [data-testid="stSidebar"] .sidebar-title-text * {{
            color: #FFFFFF !important;
        }}
        
        [data-testid="stSidebar"] hr {{
            border-color: rgba(255, 255, 255, 0.2) !important;
        }}
        
        .stRadio > div {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        
        .stRadio > div > label > div[data-baseweb="radio"] {{
            display: none !important;
        }}
        
        .stRadio > div > label > div:first-child {{
            display: none !important;
        }}
        
        [data-testid="stSidebar"] .stRadio > div > label {{
            padding: 10px 12px !important;
            border-radius: 6px !important;
            transition: all 0.2s !important;
            cursor: pointer !important;
            background-color: transparent !important;
            color: rgba(255, 255, 255, 0.8) !important;
            border: none !important;
            margin: 0 !important;
            font-size: 14px !important;
            display: flex !important;
            align-items: center !important;
        }}
        
        [data-testid="stSidebar"] .stRadio > div > label:hover {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: #FFFFFF !important;
        }}
        
        [data-testid="stSidebar"] .stRadio > div > label[data-testid*="selected"] {{
            background-color: rgba(255, 255, 255, 0.15) !important;
            border-left: 4px solid #FFFFFF !important;
            padding-left: 8px !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }}
        
        .stRadio > div > label > span {{
            margin-left: 0 !important;
        }}
        
        h1, h2, h3 {{
            color: #1F2937;
            letter-spacing: -0.5px;
            margin-top: 0;
        }}
        
        .result-card {{
            background: {BRAND_COLORS['white']};
            padding: 28px;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            font-family: 'Arita-Sans-Medium', 'Arita-Dotum-Medium', sans-serif;
        }}

        .badge-high, .badge-mid, .badge-low {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 14px;
        }}

        .badge-high {{ background: #dcfce7; color: #166534; }}
        .badge-mid  {{ background: #fef9c3; color: #854d0e; }}
        .badge-low  {{ background: #fee2e2; color: #991b1b; }}

        .muted {{ color: #6b7280; }}
        .hr {{ border: none; height: 1px; background: #E5E7EB; margin: 18px 0; }}
        .h1 {{ font-size: 40px; margin: 8px 0 6px; font-weight: 800; }}
        .h2 {{ font-size: 26px; margin: 0 0 6px; font-weight: 800; }}
        .h4 {{ font-size: 16px; margin: 0 0 6px; font-weight: 800; }}
        .small {{ color: #6b7280; font-size: 13px; line-height: 1.45; }}

        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        .streamlit-expanderHeader {{
            background-color: {BRAND_COLORS['white']};
            border-radius: 8px;
            border: 1px solid #E5E7EB;
        }}
        
        div[data-testid="stExpander"] details {{
            background: #FFFFFF;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 14px;
            padding: 10px 14px;
        }}
        
        div[data-testid="stExpander"] summary {{
            font-weight: 600;
            color: #001C58;
            font-size: 0.98rem;
        }}
        
        div[data-testid="stExpander"] .stMarkdown {{
            margin-top: 6px;
        }}

        div[data-testid="stExpander"] > details > summary:hover {{
            color: #1F2937 !important;
        }}
        div[data-testid="stExpander"] > details > summary:hover svg {{
            fill: #1F2937 !important;
        }}

        
        /* 이미지 타입 가이드 스타일 */
        .ap-row {{
            display: flex;
            gap: 12px;
            align-items: flex-start;
            padding: 10px 8px;
            border-radius: 12px;
            transition: background-color 0.2s;
        }}
        
        .ap-row:hover {{
            background: rgba(31,87,149,0.06);
        }}
        
        .ap-row b {{
            color: #111827;
        }}
        
        .ap-k {{
            width: 92px;
            flex: 0 0 92px;
        }}
        
        .ap-v {{
            flex: 1;
        }}
        
        .ap-chip {{
            display: inline-block;
            padding: 4px 9px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(31,87,149,0.10);
            color: #1F5795;
            border: 1px solid rgba(31,87,149,0.18);
        }}
        
        .ap-small {{
            font-size: 0.88rem;
            color: #6B7280;
            margin-top: 2px;
        }}
        
        .ap-divider {{
            height: 1px;
            background: rgba(0,0,0,0.06);
            margin: 10px 0 12px 0;
        }}
        
        .ap-muted {{
            color: #6B7280;
            font-size: 0.9rem;
        }}

        div[data-baseweb="select"] > div {{
            border-radius: 8px !important;
            border: 1px solid #E5E7EB !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
            background-color: {BRAND_COLORS['white']} !important;
        }}
        
        div[data-baseweb="select"] > div:hover {{
            border-color: {BRAND_COLORS['primary']} !important;
        }}
        
        div[data-testid="stSelectbox"] {{
            max-width: 320px !important;
        }}
        
        div[data-testid="stSelectbox"] > div {{
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }}
        
        div[data-testid="stSelectbox"] [data-baseweb="select"] {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
            font-size: 13px !important;
        }}
        
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
            font-size: 13px !important;
        }}
        
        div[data-testid="stSelectbox"] span {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
            font-size: 13px !important;
        }}
        
        .stButton > button {{
            border-radius: 6px;
            border: 1px solid {BRAND_COLORS['primary']};
            background-color: {BRAND_COLORS['primary']};
            color: {BRAND_COLORS['white']};
        }}
        
        .stButton > button:hover {{
            background-color: {BRAND_COLORS['deep']};
            border-color: {BRAND_COLORS['deep']};
        }}
        
        /* 이미지 유형 가이드 카드 내부 이미지 래퍼 */
        .image-type-card-wrapper {{
            width: 100% !important;
            aspect-ratio: 1 !important;
            overflow: hidden !important;
            border-radius: 8px !important;
            margin-bottom: 12px !important;
        }}
        
        .image-type-card-wrapper div[data-testid="stImage"] {{
            margin: 0 !important;
            width: 100% !important;
            padding: 0 !important;
        }}
        
        .image-type-card-wrapper div[data-testid="stImage"] img {{
            width: 100% !important;
            height: 100% !important;
            object-fit: cover !important;
            border-radius: 8px !important;
            aspect-ratio: 1 !important;
        }}
        
        .type-guide {{
            margin: 0;
        }}
        
        /* 섹션 스타일 통일 */
        .section {{
            margin-top: 36px;
            margin-bottom: 0;
        }}
        
        .section-title,
        h4.section-title {{
            font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', sans-serif !important;
            font-size: 20px;
            font-weight: 700 !important;
            color: #111827;
            margin: 0 0 8px 0;
        }}
        
        .section-desc {{
            font-size: 14px;
            color: #6B7280;
            margin: 0 0 16px 0;
            line-height: 1.6;
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        .js-plotly-plot .gtitle,
        .plotly .gtitle,
        [class*="plotly"] .gtitle,
        div[data-testid="stPlotlyChart"] .gtitle,
        .stPlotlyChart .gtitle,
        .gtitle {{
            font-weight: 400 !important;
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        .js-plotly-plot .gtitle text,
        .plotly .gtitle text,
        [class*="plotly"] .gtitle text,
        div[data-testid="stPlotlyChart"] .gtitle text,
        .stPlotlyChart .gtitle text,
        .gtitle text,
        .gtitle tspan {{
            font-weight: 400 !important;
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        svg .gtitle text,
        svg .gtitle tspan {{
            font-weight: 400 !important;
        }}
        
        .card {{
            padding: 18px;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            background: #fff;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            margin-bottom: 24px;
        }}
        
        .type-guide-intro {{
            font-size: 14px;
            color: #4B5563;
            margin-bottom: 24px;
            line-height: 1.6;
            font-family: 'Arita-Dotum-Medium', sans-serif !important;
        }}
        
        .type-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }}
        
        .type-card {{
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 14px;
            overflow: hidden;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            display: flex;
            flex-direction: column;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            margin-bottom: 20px;
        }}
        
        .type-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
        }}
        
        .type-card-header {{
            display: flex;
            gap: 8px;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .type-chip {{
            background: rgba(31, 87, 149, 0.10);
            border: 1px solid rgba(31, 87, 149, 0.25);
            color: #1F5795;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
            white-space: nowrap;
            font-family: 'Arita-Dotum-Bold', sans-serif !important;
        }}
        
        .type-title {{
            font-size: 14px;
            font-weight: 500 !important;
            color: #111827;
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        .type-image-wrapper {{
            aspect-ratio: 1 / 1;
            height: 170px;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 10px;
            background-color: #f3f4f6;
        }}
        
        .type-image-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}
        
        .type-description {{
            font-size: 12px;
            color: #6B7280;
            margin-top: 0;
            line-height: 1.5;
            font-family: 'Arita-Dotum-Medium', sans-serif !important;
        }}
        
        @media (max-width: 1100px) {{
            .type-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
        
        @media (max-width: 720px) {{
            .type-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def apply_custom_style():
    """전역 스타일 적용"""
    inject_style()
