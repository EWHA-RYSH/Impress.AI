import streamlit as st
import pandas as pd

from utils.data_loader import load_meta_df
from utils.eda_metrics import (
    preprocess_country_data,
    get_image_type_distribution,
    get_performance_summary,
    get_top_percentile_metrics
)
from utils.metrics import (
    compute_performance_kpis,
    format_percentage,
    format_engagement_rate
)
from utils.charts import plot_usage_vs_engagement
from components.layout import (
    render_page_header,
    render_kpi_card,
    render_action_items,
    get_type_name,
    render_image_type_guide,
    section_gap
)

def render():
    df_meta = load_meta_df()
    selected_country = st.session_state.get("selected_country", sorted(df_meta["country"].unique())[0])
    df_country = preprocess_country_data(df_meta, selected_country)
    
    if len(df_country) == 0:
        st.warning(f"{selected_country}에 대한 데이터가 없습니다.")
        return
    
    # 페이지 헤더 (국가 선택기 포함)
    countries = sorted(df_meta["country"].unique())
    render_page_header(
        "성과 분석",
        countries=countries,
        selected_country=selected_country,
        n_posts=len(df_country),
        description="국가별 콘텐츠 성과 데이터를 기반으로 이미지 유형별 참여 패턴과 활용 효율을 비교하여 "
                    "성과가 높은 콘텐츠 유형과 최적화 기회를 도출합니다."
    )
    
    current_country = st.session_state.get("selected_country", selected_country)
    if current_country != selected_country:
        selected_country = current_country
        df_country = preprocess_country_data(df_meta, selected_country)
        if len(df_country) == 0:
            st.warning(f"{selected_country}에 대한 데이터가 없습니다.")
            return
    
    section_gap(16)
    with st.expander("📁 이미지 유형 기준", expanded=False):
        st.markdown(
            """
            <div style="
                font-size: 14px;
                color: #6B7280;
                line-height: 1.6;
                margin-bottom: 20px;
            ">
                Type 1~6은 게시물의 이미지 구성 방식이며, KPI 해석/성과 비교의 기준으로 사용됩니다.<br>
            </div>
            """,
            unsafe_allow_html=True
        )
        render_image_type_guide()
    
    section_gap(40)
    
    kpis = compute_performance_kpis(df_country)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if kpis['best_engagement']['type']:
            best_name = get_type_name(kpis['best_engagement']['type'])
            render_kpi_card(
                "최고 참여율 타입",
                f"{best_name}",
                subtext=f"Type {kpis['best_engagement']['type']} · 참여율: {format_engagement_rate(kpis['best_engagement']['value'])}",
                highlight=True
            )
        else:
            render_kpi_card("최고 참여율 타입", "N/A")
    
    with col2:
        if kpis['underused_opportunity']['type']:
            underused_name = get_type_name(kpis['underused_opportunity']['type'])
            render_kpi_card(
                "과소 활용 기회",
                f"{underused_name}",
                subtext=f"Type {kpis['underused_opportunity']['type']} · 높은 참여율({format_engagement_rate(kpis['underused_opportunity']['engagement'])})이나 낮은 활용도({format_percentage(kpis['underused_opportunity']['usage'])})"
            )
        else:
            render_kpi_card("과소 활용 기회", "N/A")
    
    with col3:
        stability_label = "안정적" if kpis['stability']['label'] == "Stable" else "변동적" if kpis['stability']['label'] == "Volatile" else kpis['stability']['label']
        render_kpi_card(
            "안정성",
            stability_label,
            subtext="성과 일관성"
        )
    
    section_gap(48)
    
    st.markdown(
        """
        <div class="section">
            <h4 class="section-title">고성과 달성 가능성</h4>
            <div class="section-desc">각 이미지 유형이 상위 10% 성과를 달성할 확률과 상위 성과 내에서의 집중도를 확인하여, 고성과 달성 가능성이 높은 콘텐츠 유형을 파악합니다.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    section_gap(16)
    prob_10, conc_10, threshold_10 = get_top_percentile_metrics(df_country, 10)
    
    if len(prob_10) > 0:
        best_prob_type = prob_10.loc[prob_10["p_top10"].idxmax(), "img_type"]
        best_prob_value = prob_10.loc[prob_10["p_top10"].idxmax(), "p_top10"]
        best_prob_name = get_type_name(best_prob_type)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style="
                    background-color: #FFFFFF;
                    border: 1px solid #E5E7EB;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                ">
                    <div style="font-size: 13px; color: #6B7280; margin-bottom: 8px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                        Top 10% 달성 확률 최고
                    </div>
                    <div style="font-size: 24px; font-weight: 700; color: #1F2937; margin-bottom: 4px; font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Medium', sans-serif !important;">
                        {best_prob_name}
                    </div>
                    <div style="font-size: 16px; color: #1F5795; font-weight: 600; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                        {best_prob_value*100:.1f}%
                    </div>
                    <div style="font-size: 12px; color: #9CA3AF; margin-top: 8px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                        Type {best_prob_type} · 전체 게시물 중 상위 10% 성과 달성 확률
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            if len(conc_10) > 0:
                best_conc_type = conc_10.loc[conc_10["share_in_top10"].idxmax(), "img_type"]
                best_conc_value = conc_10.loc[conc_10["share_in_top10"].idxmax(), "share_in_top10"]
                best_conc_name = get_type_name(best_conc_type)
                
                st.markdown(
                    f"""
                    <div style="
                        background-color: #FFFFFF;
                        border: 1px solid #E5E7EB;
                        border-radius: 8px;
                        padding: 20px;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                    ">
                        <div style="font-size: 13px; color: #6B7280; margin-bottom: 8px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                            Top 10% 내 집중도 최고
                        </div>
                        <div style="font-size: 24px; font-weight: 700; color: #1F2937; margin-bottom: 4px; font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Medium', sans-serif !important;">
                            {best_conc_name}
                        </div>
                        <div style="font-size: 16px; color: #1F5795; font-weight: 600; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                            {best_conc_value*100:.1f}%
                        </div>
                        <div style="font-size: 12px; color: #9CA3AF; margin-top: 8px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                            Type {best_conc_type} · 상위 10% 성과 내에서 차지하는 비율
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Top 10% 성과 데이터가 없습니다.")
    
    section_gap(48)
    
    type_count, type_ratio = get_image_type_distribution(df_country)
    perf_summary = get_performance_summary(df_country)
    
    st.markdown(
        """
        <div class="section">
            <h4 class="section-title">활용도 vs 참여율</h4>
            <div class="section-desc">활용 빈도와 참여율을 함께 비교하여, 과소 활용되었지만 성과가 높은 콘텐츠 유형을 탐색합니다.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    section_gap(16)
    plot_usage_vs_engagement(
        type_ratio,
        perf_summary,
        selected_country,
        highlight_type=kpis['best_engagement']['type']
    )
    
    section_gap(48)
    
    actions = []
    
    if kpis['underused_opportunity']['type']:
        underused_type_name = get_type_name(kpis['underused_opportunity']['type'])
        actions.append({
            "action": f"{underused_type_name} (Type {kpis['underused_opportunity']['type']}) 활용도 증가",
            "reason": f"높은 참여율({format_engagement_rate(kpis['underused_opportunity']['engagement'])})을 보이지만 현재 활용도가 {format_percentage(kpis['underused_opportunity']['usage'])}로 낮습니다."
        })
    
    from utils.eda_metrics import get_usage_vs_performance
    _, _, overused = get_usage_vs_performance(df_country, 10)
    
    if len(overused) > 0:
        overused_type = int(overused.iloc[0]["img_type"])
        overused_type_name = get_type_name(overused_type)
        overused_usage = overused.iloc[0]["usage_share"] * 100
        overused_eng = overused.iloc[0]["eng_mean"]
        actions.append({
            "action": f"{overused_type_name} (Type {overused_type}) 활용도 감소",
            "reason": f"활용도는 높지만({format_percentage(overused_usage)}) 참여율이 낮습니다({format_engagement_rate(overused_eng)}). 더 높은 성과를 보이는 타입으로 재배분을 고려하세요."
        })
    
    type_counts = type_count.to_dict()
    low_sample_types = [t for t, count in type_counts.items() if count < 10]
    if low_sample_types:
        actions.append({
            "action": "주의사항",
            "reason": f"Type {', '.join(map(str, low_sample_types))}는 샘플 크기가 작아(<10개 게시글) 결과의 신뢰성이 낮을 수 있습니다."
        })
    
    if actions:
        render_action_items(actions)
    
    with st.expander("상세 통계 보기", expanded=False):
        st.markdown("##### 이미지 유형별 평균 성과")
        perf_display = perf_summary.copy()
        perf_display.columns = [
            "이미지 타입",
            "개수",
            "평균 좋아요",
            "중앙값 좋아요",
            "평균 댓글",
            "중앙값 댓글",
            "평균 참여율",
            "중앙값 참여율"
        ]
        if "평균 참여율" in perf_display.columns:
            perf_display["평균 참여율"] = perf_display["평균 참여율"].apply(lambda x: format_engagement_rate(x))
        if "중앙값 참여율" in perf_display.columns:
            perf_display["중앙값 참여율"] = perf_display["중앙값 참여율"].apply(lambda x: format_engagement_rate(x))
        
        # HTML 테이블로 렌더링 (중앙 정렬)
        perf_rows = []
        for idx, row in perf_display.iterrows():
            bg_color = "#FFFFFF" if idx % 2 == 0 else "#FAFBFC"
            border_top = "border-top: 1px solid #E5E7EB;" if idx == 0 else ""
            
            # 숫자 포맷팅
            avg_likes = f"{row['평균 좋아요']:.1f}" if isinstance(row['평균 좋아요'], (int, float)) else str(row['평균 좋아요'])
            med_likes = f"{row['중앙값 좋아요']:.1f}" if isinstance(row['중앙값 좋아요'], (int, float)) else str(row['중앙값 좋아요'])
            avg_comments = f"{row['평균 댓글']:.1f}" if isinstance(row['평균 댓글'], (int, float)) else str(row['평균 댓글'])
            med_comments = f"{row['중앙값 댓글']:.1f}" if isinstance(row['중앙값 댓글'], (int, float)) else str(row['중앙값 댓글'])
            
            perf_rows.append(f"""
            <tr style="background-color: {bg_color};">
                <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; font-weight: 500; color: #111827; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; {border_top}">{row['이미지 타입']}</td>
                <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; color: #374151; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; {border_top}">{row['개수']}</td>
                <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; color: #374151; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; {border_top}">{avg_likes}</td>
                <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; color: #374151; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; {border_top}">{med_likes}</td>
                <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; color: #374151; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; {border_top}">{avg_comments}</td>
                <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; color: #374151; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; {border_top}">{med_comments}</td>
                <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; color: #1F5795; font-weight: 500; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; {border_top}">{row['평균 참여율']}</td>
                <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; color: #1F5795; font-weight: 500; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; {border_top}">{row['중앙값 참여율']}</td>
            </tr>
            """)
        
        perf_table_html = f"""
        <div style="
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            margin-bottom: 24px;
        ">
            <table style="
                width: 100%;
                border-collapse: collapse;
                margin: 0;
                font-size: 13px;
                font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
            ">
                <thead>
                    <tr style="background: linear-gradient(to bottom, #F9FAFB 0%, #F3F4F6 100%);">
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">이미지 타입</th>
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">개수</th>
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">평균 좋아요</th>
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">중앙값 좋아요</th>
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">평균 댓글</th>
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">중앙값 댓글</th>
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">평균 참여율</th>
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">중앙값 참여율</th>
                    </tr>
                </thead>
            <tbody>
                {''.join(perf_rows)}
            </tbody>
        </table>
        </div>
        """
        
        st.markdown(perf_table_html, unsafe_allow_html=True)
        
        st.markdown("##### 고성과 달성 가능성 (Top 10%)")
        if len(prob_10) > 0:
            prob_display = prob_10.copy()
            prob_display.columns = ["이미지 타입", "Top 10% 달성 확률"]
            prob_display["Top 10% 달성 확률"] = prob_display["Top 10% 달성 확률"].apply(lambda x: f"{x*100:.1f}%")
            
            # HTML 테이블로 렌더링
            prob_rows = []
            for idx, row in prob_display.iterrows():
                bg_color = "#FFFFFF" if idx % 2 == 0 else "#FAFBFC"
                border_top = "border-top: 1px solid #E5E7EB;" if idx == 0 else ""
                prob_rows.append(f"""
                <tr style="background-color: {bg_color};">
                    <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; font-weight: 500; color: #111827; {border_top}">{row['이미지 타입']}</td>
                    <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; color: #1F5795; font-weight: 500; {border_top}">{row['Top 10% 달성 확률']}</td>
                </tr>
                """)
            
            prob_table_html = f"""
            <div style="
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                margin-bottom: 24px;
            ">
            <table style="
                width: 100%;
                border-collapse: collapse;
                margin: 0;
                font-size: 13px;
                font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
            ">
                <thead>
                    <tr style="background: linear-gradient(to bottom, #F9FAFB 0%, #F3F4F6 100%);">
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">이미지 타입</th>
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">Top 10% 달성 확률</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(prob_rows)}
                </tbody>
            </table>
            </div>
            """
            
            st.markdown(prob_table_html, unsafe_allow_html=True)
        
        if len(conc_10) > 0:
            conc_display = conc_10.copy()
            conc_display.columns = ["이미지 타입", "Top 10% 내 비율"]
            conc_display["Top 10% 내 비율"] = conc_display["Top 10% 내 비율"].apply(lambda x: f"{x*100:.1f}%")
            
            # HTML 테이블로 렌더링
            conc_rows = []
            for idx, row in conc_display.iterrows():
                bg_color = "#FFFFFF" if idx % 2 == 0 else "#FAFBFC"
                border_top = "border-top: 1px solid #E5E7EB;" if idx == 0 else ""
                conc_rows.append(f"""
                <tr style="background-color: {bg_color};">
                    <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; font-weight: 500; color: #111827; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; {border_top}">{row['이미지 타입']}</td>
                    <td style="padding: 12px 16px; text-align: center; border-bottom: 1px solid #E5E7EB; color: #1F5795; font-weight: 500; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; {border_top}">{row['Top 10% 내 비율']}</td>
                </tr>
                """)
            
            conc_table_html = f"""
            <div style="
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                margin-bottom: 24px;
            ">
            <table style="
                width: 100%;
                border-collapse: collapse;
                margin: 0;
                font-size: 13px;
                font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
            ">
                <thead>
                    <tr style="background: linear-gradient(to bottom, #F9FAFB 0%, #F3F4F6 100%);">
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">이미지 타입</th>
                        <th style="padding: 14px 16px; text-align: center; border-bottom: 2px solid #E5E7EB; font-weight: 600; color: #374151; font-size: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">Top 10% 내 비율</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(conc_rows)}
                </tbody>
            </table>
            </div>
            """
            
            st.markdown(conc_table_html, unsafe_allow_html=True)
        
        st.caption(f"💡 Top 10% 기준선: 참여율 {threshold_10:.6f} 이상")
