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
        
        /* KPI 카드 전용 폰트 */
        .kpi-card-wrapper,
        .kpi-card-wrapper * {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;
        }}
        
        .kpi-value {{
            font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Bold', 'Arita-Dotum-Bold', 'Malgun Gothic', sans-serif !important;
        }}
        
        /* KPI 카드 스타일 - 인라인 스타일로 생성된 카드에도 적용 */
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #FFFFFF"][style*="border: 1px solid #E5E7EB"] {{
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
        
        /* 인사이트/액션 박스 폰트 */
        [class*="insight"], [class*="action"] {{
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        /* 인사이트 박스 스타일 */
        div[data-testid="stMarkdownContainer"] div[style*="background-color: #F9FAFB"][style*="border-left: 4px solid"] {{
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

        
        .stPlotlyChart {{
            background-color: {BRAND_COLORS['white']};
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            width: 100% !important;
            max-width: 100% !important;
            min-height: 300px !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }}
        
        .stPlotlyChart > div {{
            width: 100% !important;
            max-width: 100% !important;
            min-height: 300px !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }}
        
        .stPlotlyChart > div > div {{
            width: 100% !important;
            max-width: 100% !important;
            min-height: 300px !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }}
        
        /* Plotly 차트 내부 요소 */
        .stPlotlyChart iframe,
        .stPlotlyChart svg,
        .stPlotlyChart .js-plotly-plot {{
            width: 100% !important;
            max-width: 100% !important;
            min-height: 300px !important;
            display: block !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }}
        
        /* Plotly 차트가 숨겨지지 않도록 보장 */
        div[data-testid="stPlotlyChart"] {{
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
        }}
        
        div[data-testid="stPlotlyChart"] > div {{
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }}
        
        div[data-testid="stPlotlyChart"] iframe,
        div[data-testid="stPlotlyChart"] svg {{
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
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
            max-width: 100% !important;
            width: 100% !important;
        }}
        
        /* Plotly SVG와 iframe은 overflow 제한하지 않음 */
        .js-plotly-plot svg,
        .js-plotly-plot iframe {{
            max-width: 100% !important;
            width: 100% !important;
        }}
        
        [data-testid="column"] {{
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
        }}
        
        [data-testid="column"] > div {{
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
        }}
        
        /* Plotly 차트가 있는 컬럼은 차트가 컨테이너를 벗어나지 않도록 */
        [data-testid="column"] .stPlotlyChart,
        [data-testid="column"] .stPlotlyChart > div,
        [data-testid="column"] .stPlotlyChart > div > div,
        [data-testid="column"] .js-plotly-plot {{
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }}
        
        [data-testid="column"] .stPlotlyChart iframe,
        [data-testid="column"] .stPlotlyChart svg,
        [data-testid="column"] .js-plotly-plot svg,
        [data-testid="column"] .js-plotly-plot iframe {{
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }}
        
        /* 차트 컨테이너 추가 제약 */
        [data-testid="column"] div[data-testid="stPlotlyChart"] {{
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
        }}
        
        [data-testid="column"] div[data-testid="stPlotlyChart"] > div {{
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
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
            width: 100% !important;
            overflow-x: hidden !important;
            box-sizing: border-box !important;
        }}
        
        /* Plotly 차트가 메인 컨테이너를 벗어나지 않도록 */
        .main .block-container .stPlotlyChart,
        .main .block-container div[data-testid="stPlotlyChart"] {{
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }}
        
        /* 컬럼 그리드 제약 강화 */
        .stColumns {{
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }}
        
        .stColumns > div {{
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
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
        
        /* ============================================
           탭 스타일 - 처음부터 primary 색상으로 설정
           ============================================ */
        
        /* 탭 버튼 기본 스타일 - 선택되지 않은 탭은 연한 회색 */
        .stTabs button[data-baseweb="tab"],
        [data-baseweb="tab-list"] button {{
            color: #9CA3AF !important;
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
            font-size: 14px !important;
            padding: 12px 20px !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            background: transparent !important;
            transition: all 0.2s ease !important;
        }}
        
        /* 탭 호버 효과 */
        .stTabs button[data-baseweb="tab"]:hover,
        [data-baseweb="tab-list"] button:hover {{
            color: {BRAND_COLORS['primary']} !important;
            background-color: rgba(31, 87, 149, 0.05) !important;
        }}
        
        /* 선택된 탭 - primary 색상 */
        .stTabs button[data-baseweb="tab"][aria-selected="true"],
        [data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: {BRAND_COLORS['primary']} !important;
            border-bottom: 2px solid {BRAND_COLORS['primary']} !important;
            font-weight: 600 !important;
            font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        }}
        
        /* 선택되지 않은 탭 - 연한 회색 */
        .stTabs button[data-baseweb="tab"][aria-selected="false"],
        [data-baseweb="tab-list"] button[aria-selected="false"] {{
            color: #9CA3AF !important;
            border-bottom: 2px solid transparent !important;
        }}
        
        /* 탭 underline 2겹 문제 해결 */
        /* tab-border: 탭 전체 아래 얇은 회색 구분선 */
        .stTabs [data-baseweb="tab-border"] {{
            background-color: #E5E7EB !important;
            height: 1px !important;
        }}
        
        /* tab-highlight: 선택된 탭 밑 굵은 강조 바 */
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: {BRAND_COLORS['primary']} !important;
            height: 3px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def apply_custom_style():
    """전역 스타일 적용"""
    inject_style()
