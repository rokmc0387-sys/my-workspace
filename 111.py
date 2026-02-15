
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 데이터 생성 (Simulation)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI-MES 품질 관리 시스템", layout="wide")

st.sidebar.header("⚙️ 시뮬레이션 설정")
st.sidebar.info("실제 MES에서는 DB나 PLC에서 데이터를 실시간으로 가져옵니다.")

# 시뮬레이션 파라미터
target_val = st.sidebar.number_input("목표 치수 (Target)", value=100.0)
tolerance = st.sidebar.number_input("허용 오차 (±)", value=2.0)
data_count = st.sidebar.slider("샘플 데이터 수", 100, 1000, 300)
process_mean_shift = st.sidebar.slider("공정 평균 이동 (Mean Shift)", -1.0, 1.0, 0.2, step=0.1)

# 가상 데이터 생성 함수
def generate_data(n, target, tol, shift):
    # 정규분포를 따르는 가상 데이터 생성 (약간의 노이즈 포함)
    np.random.seed(42)
    mu = target + shift
    sigma = tol / 3.5  # 3.5 시그마 수준으로 가정
    data = np.random.normal(mu, sigma, n)
    
    # 시간축 생성
    times = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='min')
    
    df = pd.DataFrame({'Timestamp': times, 'Value': data})
    
    # 규격 상한/하한 (USL/LSL)
    usl = target + tol
    lsl = target - tol
    
    # 판정 (OK/NG)
    df['Status'] = df['Value'].apply(lambda x: 'NG' if x > usl or x < lsl else 'OK')
    
    return df, usl, lsl

# 데이터 로드
df_measure, USL, LSL = generate_data(data_count, target_val, tolerance, process_mean_shift)

st.title("🏭 AI-MES 품질 관리 대시보드")
st.markdown("---")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📊 1. 공정능력 분석 (Cpk)", "📈 2. 불량 산포 관리 (SPC)", "📉 3. 불량률 관리 (Pareto)"])

# -----------------------------------------------------------------------------
# Tab 1: 공정능력 분석 (Process Capability Analysis)
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("공정능력 지수 (Cp, Cpk) 분석")
    
    # 통계 계산
    mu = df_measure['Value'].mean()
    sigma = df_measure['Value'].std()
    
    # Cp, Cpk 계산
    Cp = (USL - LSL) / (6 * sigma)
    Cpu = (USL - mu) / (3 * sigma)
    Cpl = (mu - LSL) / (3 * sigma)
    Cpk = min(Cpu, Cpl)
    
    # KPI 카드 표시
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("평균 (Mean)", f"{mu:.3f}")
    col2.metric("표준편차 (Std)", f"{sigma:.3f}")
    col3.metric("Cp (잠재 능력)", f"{Cp:.3f}", delta_color="normal")
    col4.metric("Cpk (실제 능력)", f"{Cpk:.3f}", delta_color="inverse" if Cpk < 1.33 else "normal")
    
    if Cpk < 1.33:
        st.warning(f"⚠️ Cpk가 1.33 미만입니다. 공정 개선이 필요합니다. (치우침 발생)")
    else:
        st.success(f"✅ 공정이 안정적입니다.")

    # 히스토그램 및 정규분포 곡선 시각화
    fig_cpk = go.Figure()
    
    # 히스토그램
    fig_cpk.add_trace(go.Histogram(x=df_measure['Value'], nbinsx=30, name='측정값', opacity=0.7, histnorm='probability density'))
    
    # 정규분포 곡선
    x_range = np.linspace(min(df_measure['Value']), max(df_measure['Value']), 100)
    pdf = stats.norm.pdf(x_range, mu, sigma)
    fig_cpk.add_trace(go.Scatter(x=x_range, y=pdf, mode='lines', name='정규분포곡선', line=dict(color='red')))
    
    # 스펙 라인 (LSL, USL, Target)
    fig_cpk.add_vline(x=LSL, line_dash="dash", line_color="red", annotation_text="LSL")
    fig_cpk.add_vline(x=USL, line_dash="dash", line_color="red", annotation_text="USL")
    fig_cpk.add_vline(x=target_val, line_dash="dot", line_color="green", annotation_text="Target")
    
    fig_cpk.update_layout(title="공정 분포도", xaxis_title="치수", yaxis_title="밀도")
    st.plotly_chart(fig_cpk, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 2: 불량 산포 관리 (Scatter / Control Chart)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("실시간 관리도 (X-bar Style Chart)")
    
    # 관리 상한/하한 (UCL/LCL) - 통상 3시그마 사용
    ucl = mu + 3 * sigma
    lcl = mu - 3 * sigma
    
    fig_spc = go.Figure()
    
    # 전체 데이터 라인
    fig_spc.add_trace(go.Scatter(x=df_measure['Timestamp'], y=df_measure['Value'], mode='lines+markers', name='측정값'))
    
    # 규격선
    fig_spc.add_hline(y=USL, line_dash="dash", line_color="red", annotation_text="USL (규격 상한)")
    fig_spc.add_hline(y=LSL, line_dash="dash", line_color="red", annotation_text="LSL (규격 하한)")
    
    # 관리 한계선 (통계적 관리선)
    fig_spc.add_hrect(y0=lcl, y1=ucl, line_width=0, fillcolor="green", opacity=0.1, annotation_text="정상 관리 구간(±3σ)")
    
    # 이상치(Outlier) 하이라이트 (규격 벗어난 점)
    outliers = df_measure[(df_measure['Value'] > USL) | (df_measure['Value'] < LSL)]
    fig_spc.add_trace(go.Scatter(x=outliers['Timestamp'], y=outliers['Value'], mode='markers', 
                                 marker=dict(color='red', size=10, symbol='x'), name='불량(Outlier)'))

    fig_spc.update_layout(title="시계열 변동 관리도", xaxis_title="시간", yaxis_title="치수")
    st.plotly_chart(fig_spc, use_container_width=True)
    
    st.dataframe(outliers, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 3: 불량률 관리 (Defect Rate & Pareto)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("불량 유형 분석 (Pareto Chart)")
    
    # 가상 불량 유형 데이터 생성 (랜덤)
    defect_types = ['치수 불량', '스크래치', '찍힘', '이물질', '도금 불량']
    defect_counts = np.random.randint(5, 50, size=len(defect_types))
    # '치수 불량'은 위 데이터 기반으로 실제 개수 반영
    ng_count = len(df_measure[df_measure['Status'] == 'NG'])
    defect_counts[0] = ng_count 
    
    df_defect = pd.DataFrame({'Type': defect_types, 'Count': defect_counts})
    df_defect = df_defect.sort_values(by='Count', ascending=False)
    
    # 누적 비율 계산
    df_defect['Cumulative Percentage'] = df_defect['Count'].cumsum() / df_defect['Count'].sum() * 100
    
    # 파레토 차트 그리기 (Bar + Line)
    fig_pareto = go.Figure()
    
    # 막대 그래프 (빈도수)
    fig_pareto.add_trace(go.Bar(x=df_defect['Type'], y=df_defect['Count'], name='불량 수량', marker_color='indianred'))
    
    # 선 그래프 (누적 비율)
    fig_pareto.add_trace(go.Scatter(x=df_defect['Type'], y=df_defect['Cumulative Percentage'], 
                                    name='누적 점유율(%)', yaxis='y2', mode='lines+markers', line=dict(color='blue')))
    
    fig_pareto.update_layout(
        title="불량 유형별 파레토 차트",
        yaxis=dict(title='불량 수량'),
        yaxis2=dict(title='누적 점유율 (%)', overlaying='y', side='right', range=[0, 110]),
        showlegend=True
    )
    
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        st.plotly_chart(fig_pareto, use_container_width=True)
    with col_p2:
        total_defects = df_defect['Count'].sum()
        total_production = data_count
        defect_rate = (total_defects / total_production) * 100
        
        st.metric("총 생산량", f"{total_production} 개")
        st.metric("총 불량 수", f"{total_defects} 개")
        st.metric("종합 불량률", f"{defect_rate:.2f} %", delta_color="inverse")
        
        st.write("📋 **불량 유형 데이터**")
        st.dataframe(df_defect[['Type', 'Count']], hide_index=True)