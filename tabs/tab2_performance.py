import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_meta_df
from components.design_tokens import (
    get_text_style, get_bg_style, get_border_style, TEXT_COLORS, FONT_SIZES, 
    SPACING, BRAND_COLORS, FONT_WEIGHTS, FONT_FAMILIES, BORDER_RADIUS, BORDER_COLORS
)
from utils.eda_metrics import (
    preprocess_country_data,
    get_image_type_distribution,
    get_performance_summary,
    get_top_percentile_metrics,
    get_stability_metrics,
    get_response_characteristics,
    get_usage_vs_performance
)
from utils.metrics import (
    compute_performance_kpis,
    format_percentage,
    format_engagement_rate
)
from utils.charts import plot_usage_vs_engagement, apply_chart_style, BRAND_COLORS, CHART_PALETTE, LIGHT_BLUE_HIGHLIGHT, DEFAULT_BAR_COLOR, MEDIAN_COLOR, MEAN_COLOR
from utils.insights_store import load_tab_insights
from components.layout import (
    render_page_header,
    render_kpi_card,
    render_action_items,
    render_insight_bullets,
    get_type_name,
    render_image_type_guide,
    section_gap
)

def render():
    # JSON 인사이트 로드
    insights = load_tab_insights("tab2")
    
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
            f"""
            <div style="{get_text_style('md', 'tertiary')} line-height: 1.6; margin-bottom: {SPACING['xl']};">
                Type 1~6은 게시물의 이미지 구성 방식이며, KPI 해석/성과 비교의 기준으로 사용됩니다.<br>
            </div>
            """,
            unsafe_allow_html=True
        )
        render_image_type_guide()
    
    section_gap(24)
    
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
    
    type_count, type_ratio = get_image_type_distribution(df_country)
    
    # 4개 탭으로 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "성과 비교・반응 성격",
        "고성과 분석",
        "안정성 분석",
        "전략 인사이트"
    ])
    
    # ============================================
    # 탭 1: 성과 비교・반응 성격
    # ============================================
    with tab1:
        perf_summary = get_performance_summary(df_country)
        response_char = get_response_characteristics(df_country)
        
        # 참여율 분포
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">참여율 분포</h4>
                <div class="section-desc">이미지 타입별 참여율(Engagement Rate) 분포를 비교합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        if len(perf_summary) > 0:
            # Top 1만 연한 블루로 강조
            max_idx = perf_summary["eng_mean"].idxmax()
            colors = []
            text_values = []
            for idx, row in perf_summary.iterrows():
                if idx == max_idx:
                    colors.append(LIGHT_BLUE_HIGHLIGHT)  # Top 1만 연한 블루
                else:
                    colors.append(DEFAULT_BAR_COLOR)  # 나머지는 #E1E4EA
                # 값 라벨 추가 (참여율은 소수점 표시)
                text_values.append(f"{row['eng_mean']:.4f}")
            
            fig = px.bar(
                perf_summary,
                x="img_type",
                y="eng_mean",
                labels={"img_type": "이미지 타입", "eng_mean": ""},
                title="이미지 타입별 평균 참여율",
                text=text_values
            )
            fig.update_traces(
                marker_color=colors, 
                width=0.6,
                textposition="outside",
                textfont=dict(size=11, color="#6B7280", family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif")
            )
            fig = apply_chart_style(fig)
            fig.update_layout(
                bargap=0.4, 
                showlegend=False, 
                height=400,
                yaxis=dict(title=None),
                margin=dict(l=40, r=20, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        section_gap(48)
        
        # 좋아요/댓글 수 분포
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">좋아요・댓글 분포</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        col1, col2 = st.columns(2)
        with col1:
            if len(perf_summary) > 0:
                # 평균과 중앙값 모두 막대로 표시 (댓글 수 차트와 동일)
                fig1 = px.bar(
                    perf_summary,
                    x="img_type",
                    y=["likes_mean", "likes_median"],
                    labels={"img_type": "이미지 타입", "value": "", "variable": ""},
                    title="좋아요 수",
                    barmode="group",
                    color_discrete_map={"likes_mean": MEAN_COLOR, "likes_median": MEDIAN_COLOR}
                )
                # 평균은 진한 회색, 중앙값은 연한 회색 (댓글 수 차트와 동일)
                if len(fig1.data) >= 2:
                    fig1.data[0].marker.color = MEAN_COLOR  # 평균 - #9CA3AF
                    fig1.data[0].name = "평균"
                    fig1.data[1].marker.color = MEDIAN_COLOR  # 중앙값 - #E5E7EB
                    fig1.data[1].name = "중앙값"
                fig1.update_traces(width=0.6)
                fig1 = apply_chart_style(fig1)
                fig1.update_layout(
                    bargap=0.4, 
                    height=400,
                    showlegend=True,
                    yaxis=dict(title=None),
                    margin=dict(l=40, r=40, t=40, b=60),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.15,
                        xanchor="left",
                        x=0,
                        font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                        itemwidth=30,
                        tracegroupgap=5,
                        itemsizing="constant",
                        bgcolor="rgba(255,255,255,0)",
                        bordercolor="rgba(255,255,255,0)"
                    )
                )
                st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
        
        with col2:
            if len(perf_summary) > 0:
                # 댓글 수는 둘 다 막대 유지, 색 대비 더 벌리기
                fig2 = px.bar(
                    perf_summary,
                    x="img_type",
                    y=["comments_mean", "comments_median"],
                    labels={"img_type": "이미지 타입", "value": "", "variable": ""},
                    title="댓글 수",
                    barmode="group",
                    color_discrete_map={"comments_mean": CHART_PALETTE[2], "comments_median": CHART_PALETTE[6]}
                )
                # 평균은 #9CA3AF, 중앙값은 #E5E7EB
                if len(fig2.data) >= 2:
                    fig2.data[0].marker.color = MEAN_COLOR  # 평균 - #9CA3AF
                    fig2.data[0].name = "평균"
                    fig2.data[1].marker.color = MEDIAN_COLOR  # 중앙값 - #E5E7EB
                    fig2.data[1].name = "중앙값"
                fig2.update_traces(width=0.5)  # 막대 폭 약간 줄이기
                fig2 = apply_chart_style(fig2)
                fig2.update_layout(
                    bargap=0.4, 
                    height=400,
                    showlegend=True,
                    yaxis=dict(title=None),
                    margin=dict(l=40, r=40, t=40, b=60),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.15,
                        xanchor="left",
                        x=0,
                        font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                        itemwidth=30,
                        tracegroupgap=5,
                        itemsizing="constant",
                        bgcolor="rgba(255,255,255,0)",
                        bordercolor="rgba(255,255,255,0)"
                    )
                )
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        
        section_gap(48)
        
        # 댓글 비율 분포
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">댓글 비율 분포</h4>
                <div class="section-desc">이미지 타입별 댓글 비율을 비교합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        if len(response_char) > 0:
            # Top 1만 연한 블루로 강조
            max_idx = response_char["comment_ratio_mean"].idxmax()
            colors = []
            text_values = []
            for idx, row in response_char.iterrows():
                if idx == max_idx:
                    colors.append(LIGHT_BLUE_HIGHLIGHT)  # Top 1만 연한 블루
                else:
                    colors.append(DEFAULT_BAR_COLOR)  # 나머지는 #E1E4EA
                # 값 라벨 추가 (퍼센트)
                text_values.append(f"{row['comment_ratio_mean']*100:.1f}%")
            
            fig4 = px.bar(
                response_char,
                x="img_type",
                y="comment_ratio_mean",
                labels={"img_type": "이미지 타입", "comment_ratio_mean": ""},
                title="이미지 타입별 평균 댓글 비율",
                text=text_values
            )
            fig4.update_traces(
                marker_color=colors, 
                width=0.6,
                textposition="outside",
                textfont=dict(size=11, color="#6B7280", family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif")
            )
            fig4 = apply_chart_style(fig4)
            fig4.update_layout(
                bargap=0.4, 
                showlegend=False, 
                height=400,
                yaxis=dict(title=None),
                margin=dict(l=40, r=20, t=40, b=40)
            )
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        
        # 국가별 인사이트 표시
        country_insight = insights.get(selected_country, {})
        performance_bullets = country_insight.get("performance_comparison", {}).get("bullets", [])
        if performance_bullets:
            section_gap(24)
            render_insight_bullets(performance_bullets, title="국가별 인사이트")
        
        # 상세 통계 보기
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
            st.dataframe(perf_display, use_container_width=True, hide_index=True)
            
            if len(response_char) > 0:
                st.markdown("##### 댓글 비율 통계")
                response_display = response_char.copy()
                response_display.columns = [
                    "이미지 타입",
                    "개수",
                    "평균 댓글 비율",
                    "중앙값 댓글 비율",
                    "평균 댓글 수",
                    "평균 좋아요 수"
                ]
                st.dataframe(response_display, use_container_width=True, hide_index=True)
    
    # ============================================
    # 탭 2: 고성과 분석
    # ============================================
    with tab2:
        prob_10, conc_10, threshold_10 = get_top_percentile_metrics(df_country, 10)
        prob_30, conc_30, threshold_30 = get_top_percentile_metrics(df_country, 30)
        
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">고성과 달성 가능성</h4>
                <div class="section-desc">각 이미지 유형이 상위 10% 및 30% 성과를 달성할 확률과 상위 성과 내에서의 집중도를 확인하여, 고성과 달성 가능성이 높은 콘텐츠 유형을 파악합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        col1, col2 = st.columns(2)
        
        # Top 10% 박스
        with col1:
            if len(prob_10) > 0 and len(conc_10) > 0:
                best_prob_type = prob_10.loc[prob_10["p_top10"].idxmax(), "img_type"]
                best_prob_value = prob_10.loc[prob_10["p_top10"].idxmax(), "p_top10"]
                best_prob_name = get_type_name(best_prob_type)
                
                best_conc_type = conc_10.loc[conc_10["share_in_top10"].idxmax(), "img_type"]
                best_conc_value = conc_10.loc[conc_10["share_in_top10"].idxmax(), "share_in_top10"]
                best_conc_name = get_type_name(best_conc_type)
                
                st.markdown(
                    f"""
                    <div class="kpi-card-wrapper" style="{get_bg_style('white')} {get_border_style('default')} border-radius: {BORDER_RADIUS['md']}; padding: {SPACING['xl']}; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                        <div style="background: rgba(31, 87, 149, 0.10); border: 1px solid rgba(31, 87, 149, 0.25); color: {BRAND_COLORS['primary']}; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; white-space: nowrap; font-family: 'Arita-Dotum-Bold', sans-serif !important; display: inline-block; margin-bottom: {SPACING['lg']};">
                            Top 10%
                        </div>
                        <div style="margin-bottom: {SPACING['xl']};">
                            <div style="{get_text_style('sm', 'tertiary', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']};">
                                달성 확률 최고
                            </div>
                            <div style="font-size: 24px !important; font-weight: 900 !important; color: {BRAND_COLORS['primary']} !important; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']}; line-height: 1.2;">
                                {best_prob_name}
                            </div>
                            <div style="{get_text_style('lg', 'accent', 'semibold', family='bold')} font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['sm']};">
                                {best_prob_value*100:.1f}%
                            </div>
                            <div style="{get_text_style('xs', 'muted', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                Type {best_prob_type} · 전체 게시물 중 상위 10% 성과 달성 확률
                            </div>
                        </div>
                        <div style="border-top: 1px solid {BORDER_COLORS['light']}; padding-top: {SPACING['lg']};">
                            <div style="{get_text_style('sm', 'tertiary', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']};">
                                집중도 최고
                            </div>
                            <div style="font-size: 24px !important; font-weight: 900 !important; color: {BRAND_COLORS['primary']} !important; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']}; line-height: 1.2;">
                                {best_conc_name}
                            </div>
                            <div style="{get_text_style('lg', 'accent', 'semibold', family='bold')} font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['sm']};">
                                {best_conc_value*100:.1f}%
                            </div>
                            <div style="{get_text_style('xs', 'muted', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                Type {best_conc_type} · 상위 10% 성과 내에서 차지하는 비율
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Top 10% 성과 데이터가 없습니다.")
        
        # Top 30% 박스
        with col2:
            if len(prob_30) > 0 and len(conc_30) > 0:
                best_prob30_type = prob_30.loc[prob_30["p_top30"].idxmax(), "img_type"]
                best_prob30_value = prob_30.loc[prob_30["p_top30"].idxmax(), "p_top30"]
                best_prob30_name = get_type_name(best_prob30_type)
                
                best_conc30_type = conc_30.loc[conc_30["share_in_top30"].idxmax(), "img_type"]
                best_conc30_value = conc_30.loc[conc_30["share_in_top30"].idxmax(), "share_in_top30"]
                best_conc30_name = get_type_name(best_conc30_type)
                
                st.markdown(
                    f"""
                    <div class="kpi-card-wrapper" style="{get_bg_style('white')} {get_border_style('default')} border-radius: {BORDER_RADIUS['md']}; padding: {SPACING['xl']}; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                        <div style="background: rgba(31, 87, 149, 0.10); border: 1px solid rgba(31, 87, 149, 0.25); color: {BRAND_COLORS['primary']}; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; white-space: nowrap; font-family: 'Arita-Dotum-Bold', sans-serif !important; display: inline-block; margin-bottom: {SPACING['lg']};">
                            Top 30%
                        </div>
                        <div style="margin-bottom: {SPACING['xl']};">
                            <div style="{get_text_style('sm', 'tertiary', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']};">
                                달성 확률 최고
                            </div>
                            <div style="font-size: 24px !important; font-weight: 900 !important; color: {BRAND_COLORS['primary']} !important; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']}; line-height: 1.2;">
                                {best_prob30_name}
                            </div>
                            <div style="{get_text_style('lg', 'accent', 'semibold', family='bold')} font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['sm']};">
                                {best_prob30_value*100:.1f}%
                            </div>
                            <div style="{get_text_style('xs', 'muted', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                Type {best_prob30_type} · 전체 게시물 중 상위 30% 성과 달성 확률
                            </div>
                        </div>
                        <div style="border-top: 1px solid {BORDER_COLORS['light']}; padding-top: {SPACING['lg']};">
                            <div style="{get_text_style('sm', 'tertiary', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']};">
                                집중도 최고
                            </div>
                            <div style="font-size: 24px !important; font-weight: 900 !important; color: {BRAND_COLORS['primary']} !important; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']}; line-height: 1.2;">
                                {best_conc30_name}
                            </div>
                            <div style="{get_text_style('lg', 'accent', 'semibold', family='bold')} font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['sm']};">
                                {best_conc30_value*100:.1f}%
                            </div>
                            <div style="{get_text_style('xs', 'muted', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                Type {best_conc30_type} · 상위 30% 성과 내에서 차지하는 비율
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Top 30% 성과 데이터가 없습니다.")
        
        section_gap(48)
        
        # Top 10% vs Top 30% 비교 차트
        if len(prob_10) > 0 and len(prob_30) > 0:
            st.markdown(
                """
                <div class="section">
                    <h4 class="section-title">Top 10% vs Top 30% 달성 확률 비교</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            section_gap(16)
            
            comparison_df = pd.DataFrame({
                "img_type": prob_10["img_type"],
                "Top 10%": prob_10["p_top10"],
                "Top 30%": prob_30["p_top30"]
            })
            
            fig = px.bar(
                comparison_df,
                x="img_type",
                y=["Top 10%", "Top 30%"],
                labels={"img_type": "이미지 타입", "value": "", "variable": "기준"},
                title="이미지 타입별 고성과 달성 확률",
                barmode="group",
                color_discrete_map={"Top 10%": MEAN_COLOR, "Top 30%": MEDIAN_COLOR}
            )
            # Top 10%는 #9CA3AF, Top 30%는 #E5E7EB
            if len(fig.data) >= 2:
                fig.data[0].marker.color = MEAN_COLOR  # Top 10% - #9CA3AF
                fig.data[0].name = "Top 10%"
                fig.data[1].marker.color = MEDIAN_COLOR  # Top 30% - #E5E7EB
                fig.data[1].name = "Top 30%"
            # 모든 막대 너비 통일 (더 작게 조정)
            fig.update_traces(width=0.4)
            fig = apply_chart_style(fig)
            fig.update_layout(
                bargap=0.4, 
                height=400,
                showlegend=True,
                yaxis=dict(title=None),
                margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                    itemwidth=30,
                    tracegroupgap=5,
                    itemsizing="constant",
                    bgcolor="rgba(255,255,255,0)",
                    bordercolor="rgba(255,255,255,0)"
                )
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        # 고성과 분석 인사이트 표시
        country_insight = insights.get(selected_country, {})
        high_perf_insight = country_insight.get("high_performance_analysis", {})
        if high_perf_insight:
            section_gap(48)
            summary = high_perf_insight.get("summary", "")
            bullets = high_perf_insight.get("bullets", [])
            
            # 요약 문장을 bullets 앞에 추가하여 박스 안에 표시
            all_bullets = []
            if summary:
                all_bullets.append(f"👉 {summary}")
            if bullets:
                all_bullets.extend(bullets)
            
            if all_bullets:
                render_insight_bullets(all_bullets, title="고성과 분석")
        
        # 상세 통계 보기
        with st.expander("상세 통계 보기", expanded=False):
            st.markdown("##### Top 10% 달성 확률")
            if len(prob_10) > 0:
                prob_display = prob_10.copy()
                prob_display.columns = ["이미지 타입", "Top 10% 달성 확률"]
                prob_display["Top 10% 달성 확률"] = prob_display["Top 10% 달성 확률"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(prob_display, use_container_width=True, hide_index=True)
            
            if len(conc_10) > 0:
                conc_display = conc_10.copy()
                conc_display.columns = ["이미지 타입", "Top 10% 내 비율"]
                conc_display["Top 10% 내 비율"] = conc_display["Top 10% 내 비율"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(conc_display, use_container_width=True, hide_index=True)
            
            st.caption(f"💡 Top 10% 기준선: 참여율 {threshold_10:.6f} 이상")
            
            st.markdown("##### Top 30% 달성 확률")
            if len(prob_30) > 0:
                prob30_display = prob_30.copy()
                prob30_display.columns = ["이미지 타입", "Top 30% 달성 확률"]
                prob30_display["Top 30% 달성 확률"] = prob30_display["Top 30% 달성 확률"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(prob30_display, use_container_width=True, hide_index=True)
            
            if len(conc_30) > 0:
                conc30_display = conc_30.copy()
                conc30_display.columns = ["이미지 타입", "Top 30% 내 비율"]
                conc30_display["Top 30% 내 비율"] = conc30_display["Top 30% 내 비율"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(conc30_display, use_container_width=True, hide_index=True)
            
            st.caption(f"💡 Top 30% 기준선: 참여율 {threshold_30:.6f} 이상")
    
    # ============================================
    # 탭 3: 안정성 분석
    # ============================================
    with tab3:
        stability = get_stability_metrics(df_country)
        
        st.markdown(
            f"""
            <div class="section" style="margin-bottom: 8px;">
                <h4 class="section-title">성과 안정성 분석</h4>
                <div class="section-desc" style="margin-bottom: 0;">표준편차(STD), IQR(사분위수 범위), 변동계수(CV)를 통해 이미지 타입별 성과의 변동성과 안정성을 측정합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(24)

         #안정성 인사이트 표시
        stability_bullets = country_insight.get("stability_analysis", {}).get("bullets", [])

        if stability_bullets:
            section_gap(32)
            render_insight_bullets(
                stability_bullets)

        #그래프 표시    
        if len(stability) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 8px;">
                        <div style="{get_text_style('md', 'secondary', 'semibold')} margin-bottom: 2px;">표준편차 (STD)</div>
                        <div style="{get_text_style('sm', 'tertiary')}">성과 변동성 측정</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # 최고값 막대는 연한 하늘색으로 강조
                max_idx = stability["eng_std"].idxmax()
                colors = []
                for idx, row in stability.iterrows():
                    if idx == max_idx:
                        colors.append(LIGHT_BLUE_HIGHLIGHT)  # 최고값은 연한 하늘색
                    else:
                        colors.append(DEFAULT_BAR_COLOR)  # 나머지는 #E1E4EA
                
                fig1 = px.bar(
                    stability,
                    x="img_type",
                    y="eng_std",
                    labels={"img_type": "이미지 타입", "eng_std": ""},
                    title=None
                )
                fig1.update_traces(marker_color=colors, width=0.6)
                fig1 = apply_chart_style(fig1)
                fig1.update_layout(
                    bargap=0.4, 
                    showlegend=False, 
                    height=300,
                    yaxis=dict(title=None),
                    margin=dict(l=40, r=20, t=20, b=40),
                    title=dict(text=""),
                    xaxis=dict(title=None)
                )
                st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
            
            with col2:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 8px;">
                        <div style="{get_text_style('md', 'secondary', 'semibold')} margin-bottom: 2px;">IQR (사분위수 범위)</div>
                        <div style="{get_text_style('sm', 'tertiary')}">중간 50% 퍼짐 정도</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # 최고값 막대는 연한 하늘색으로 강조
                max_idx = stability["eng_iqr"].idxmax()
                colors = []
                for idx, row in stability.iterrows():
                    if idx == max_idx:
                        colors.append(LIGHT_BLUE_HIGHLIGHT)  # 최고값은 연한 하늘색
                    else:
                        colors.append(DEFAULT_BAR_COLOR)  # 나머지는 #E1E4EA
                
                fig2 = px.bar(
                    stability,
                    x="img_type",
                    y="eng_iqr",
                    labels={"img_type": "이미지 타입", "eng_iqr": ""},
                    title=None
                )
                fig2.update_traces(marker_color=colors, width=0.6)
                fig2 = apply_chart_style(fig2)
                fig2.update_layout(
                    bargap=0.4, 
                    showlegend=False, 
                    height=300,
                    yaxis=dict(title=None),
                    margin=dict(l=40, r=20, t=20, b=40),
                    title=dict(text=""),
                    xaxis=dict(title=None)
                )
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            
            with col3:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 8px;">
                        <div style="{get_text_style('md', 'secondary', 'semibold')} margin-bottom: 2px;">변동계수 (CV)</div>
                        <div style="{get_text_style('sm', 'tertiary')}">상대적 변동성</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # 최고값 막대는 연한 하늘색으로 강조
                max_idx = stability["eng_cv"].idxmax()
                colors = []
                for idx, row in stability.iterrows():
                    if idx == max_idx:
                        colors.append(LIGHT_BLUE_HIGHLIGHT)  # 최고값은 연한 하늘색
                    else:
                        colors.append(DEFAULT_BAR_COLOR)  # 나머지는 #E1E4EA
                
                fig3 = px.bar(
                    stability,
                    x="img_type",
                    y="eng_cv",
                    labels={"img_type": "이미지 타입", "eng_cv": ""},
                    title=None
                )
                fig3.update_traces(marker_color=colors, width=0.6)
                fig3 = apply_chart_style(fig3)
                fig3.update_layout(
                    bargap=0.4, 
                    showlegend=False, 
                    height=300,
                    yaxis=dict(title=None),
                    margin=dict(l=40, r=20, t=20, b=40),
                    title=dict(text=""),
                    xaxis=dict(title=None)
                )
                st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        
                country_insight = insights.get(selected_country, {})


        # 상세 통계 보기
        with st.expander("상세 통계 보기", expanded=False):
            if len(stability) > 0:
                stability_display = stability.copy()
                stability_display.columns = [
                    "이미지 타입",
                    "개수",
                    "평균 참여율",
                    "표준편차 (STD)",
                    "IQR",
                    "변동계수 (CV)"
                ]
                st.dataframe(stability_display, use_container_width=True, hide_index=True)
    
    # ============================================
    # 탭 4: 전략 인사이트
    # ============================================
    with tab4:
        usage_vs_perf, underused, overused = get_usage_vs_performance(df_country, 10)
        
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">활용도 vs 성과 분석</h4>
                <div class="section-desc">활용 빈도와 참여율을 함께 비교하여, 과소 활용되었지만 성과가 높은 콘텐츠 유형을 탐색합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        perf_summary = get_performance_summary(df_country)
        plot_usage_vs_engagement(
            type_ratio,
            perf_summary,
            selected_country
        )
        
        section_gap(48)
        
        # 과소 활용 타입 (확대 후보)
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">과소 활용 타입 (확대 후보)</h4>
                <div class="section-desc">높은 성과를 보이지만 활용도가 낮은 타입으로, 확대를 고려할 수 있습니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        if len(underused) > 0:
            for idx, row in underused.iterrows():
                type_num = int(row["img_type"])
                type_name = get_type_name(type_num)
                usage_pct = row["usage_share"] * 100
                eng_rate = row["eng_mean"]
                prob_top10 = row.get("p_top10", 0) * 100
                
                st.markdown(
                    f"""
                    <div style="{get_bg_style('white')} border: 1px solid #E5E7EB; border-radius: 8px; padding: {SPACING['xl']}; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: {SPACING['md']};">
                        <div style="{get_text_style('lg', 'primary', family='bold')} margin-bottom: {SPACING['xs']};">
                            {type_name} (Type {type_num})
                        </div>
                        <div style="{get_text_style('base', 'tertiary')} margin-top: {SPACING['sm']};">
                            활용도: {format_percentage(usage_pct)} · 참여율: {format_engagement_rate(eng_rate)} · Top 10% 확률: {prob_top10:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("과소 활용 타입이 없습니다.")
        
        section_gap(48)
        
        # 과대 활용 타입 (축소/개선 후보)
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">과대 활용 타입 (축소/개선 후보)</h4>
                <div class="section-desc">활용도는 높지만 성과가 낮은 타입으로, 축소하거나 개선을 고려할 수 있습니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        if len(overused) > 0:
            for idx, row in overused.iterrows():
                type_num = int(row["img_type"])
                type_name = get_type_name(type_num)
                usage_pct = row["usage_share"] * 100
                eng_rate = row["eng_mean"]
                prob_top10 = row.get("p_top10", 0) * 100
                
                st.markdown(
                    f"""
                    <div style="{get_bg_style('white')} border: 1px solid #E5E7EB; border-radius: 8px; padding: {SPACING['xl']}; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: {SPACING['md']};">
                        <div style="{get_text_style('lg', 'primary', family='bold')} margin-bottom: {SPACING['xs']};">
                            {type_name} (Type {type_num})
                        </div>
                        <div style="{get_text_style('base', 'tertiary')} margin-top: {SPACING['sm']};">
                            활용도: {format_percentage(usage_pct)} · 참여율: {format_engagement_rate(eng_rate)} · Top 10% 확률: {prob_top10:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("과대 활용 타입이 없습니다.")
        
        section_gap(48)
        
        # Action Items
        actions = []
        
        if kpis['underused_opportunity']['type']:
            underused_type_name = get_type_name(kpis['underused_opportunity']['type'])
            actions.append({
                "action": f"{underused_type_name} (Type {kpis['underused_opportunity']['type']}) 활용도 증가",
                "reason": f"높은 참여율({format_engagement_rate(kpis['underused_opportunity']['engagement'])})을 보이지만 현재 활용도가 {format_percentage(kpis['underused_opportunity']['usage'])}로 낮습니다."
            })
        
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
    
    section_gap(48)
