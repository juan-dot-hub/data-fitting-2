import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
import matplotlib.font_manager as fm
import warnings
 
# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')
 
# matplotlib 한글 폰트 및 스타일 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
font_prop = fm.FontProperties(family='DejaVu Sans')
 
# 케플러 방정식: E - e * sin(E) - M = 0
def kepler_equation(E, M, e):
    return E - e * np.sin(E) - M
 
# 케플러 방정식을 풀어 이심근점 이 E를 계산
def solve_kepler(M, e):
    E_guess = M
    E = fsolve(kepler_equation, E_guess, args=(M, e))[0]
    return E
 
# 평균 이 M 계산
def mean_anomaly(t, P, t0):
    return 2 * np.pi * (t - t0) / P
 
# 진정한 이 $nu$를 계산
def true_anomaly(E, e):
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
 
# 두 데이터 세트를 함께 피팅하기 위한 모델 함수
def combined_radial_velocity_model(t, gamma, K1, P, e, omega1, t01, K2, omega2, t02):
    # 첫 번째 데이터 세트의 인덱스를 찾기
    split_index = len(st.session_state.t_data1_session)
    t1 = t[:split_index]
    t2 = t[split_index:]
 
    # 첫 번째 데이터 세트의 시선속도 계산
    vr1 = np.zeros_like(t1)
    for i, time in enumerate(t1):
        M = mean_anomaly(time, P, t01)
        E = solve_kepler(M, e)
        theta = true_anomaly(E, e)
        vr1[i] = gamma + K1 * (np.cos(theta + omega1) + e * np.cos(omega1))
 
    # 두 번째 데이터 세트의 시선속도 계산
    vr2 = np.zeros_like(t2)
    for i, time in enumerate(t2):
        M = mean_anomaly(time, P, t02)
        E = solve_kepler(M, e)
        theta = true_anomaly(E, e)
        vr2[i] = gamma + K2 * (np.cos(theta + omega2) + e * np.cos(omega2))
 
    # 두 결과를 합쳐서 반환
    return np.concatenate((vr1, vr2))
 
# 피팅 및 결과 출력 함수
def fit_and_plot(t_data1, vr_observed1, t_data2, vr_observed2):
    # 데이터 세션 상태에 저장하여 combined_radial_velocity_model 함수에서 접근할 수 있도록 함
    st.session_state.t_data1_session = t_data1
    
    # 두 데이터 세트의 시간 및 시선속도 데이터를 결합
    t_combined = np.concatenate((t_data1, t_data2))
    vr_combined = np.concatenate((vr_observed1, vr_observed2))
 
    # 초기 추정값 설정 (gamma, K1, P, e, omega1, t01, K2, omega2, t02)
    initial_guess = [
        np.mean(vr_combined),
        (np.max(vr_observed1) - np.min(vr_observed1)) / 2, # K1
        40, # P
        0.4, # e
        0.5 * np.pi, # omega1
        0.0, # t01
        (np.max(vr_observed2) - np.min(vr_observed2)) / 2, # K2
        1.5 * np.pi, # omega2
        0.0 # t02
    ]
    # 매개변수 경계 설정
    bounds = (
        [-100, 0, 1, 0, 0, -50, 0, 0, -50],
        [100, 100, 200, 0.99, 2 * np.pi, 50, 100, 2 * np.pi, 50]
    )
 
    popt, pcov = curve_fit(
        combined_radial_velocity_model, t_combined, vr_combined,
        p0=initial_guess, bounds=bounds, maxfev=5000
    )
    
    # 피팅된 매개변수 언팩
    fitted_gamma, fitted_K1, fitted_P, fitted_e, fitted_omega1, fitted_t01, \
    fitted_K2, fitted_omega2, fitted_t02 = popt
 
    # 결과 출력
    st.write("### 공통 피팅 결과")
    st.write(f"γ = {fitted_gamma:.2f} ± {np.sqrt(pcov[0, 0]):.2f} km/s")
    st.write(f"P = {fitted_P:.2f} ± {np.sqrt(pcov[2, 2]):.2f} days")
    st.write(f"e = {fitted_e:.3f} ± {np.sqrt(pcov[3, 3]):.3f}")
    
    st.write("### 데이터 세트 1 개별 결과")
    st.write(f"K = {fitted_K1:.2f} ± {np.sqrt(pcov[1, 1]):.2f} km/s")
    st.write(f"ω = {fitted_omega1:.3f} ± {np.sqrt(pcov[4, 4]):.3f} rad")
    st.write(f"t₀ = {fitted_t01:.2f} ± {np.sqrt(pcov[5, 5]):.2f} days")
    
    st.write("### 데이터 세트 2 개별 결과")
    st.write(f"K = {fitted_K2:.2f} ± {np.sqrt(pcov[6, 6]):.2f} km/s")
    st.write(f"ω = {fitted_omega2:.3f} ± {np.sqrt(pcov[7, 7]):.3f} rad")
    st.write(f"t₀ = {fitted_t02:.2f} ± {np.sqrt(pcov[8, 8]):.2f} days")
    
    # 모델 곡선을 부드럽게 그리기 위해 더 촘촘한 시간 배열 생성
    t_model1 = np.linspace(min(t_data1), max(t_data1), 500)
    t_model2 = np.linspace(min(t_data2), max(t_data2), 500)
    
    # 각 데이터 세트에 대해 모델 곡선 계산
    vr_model1 = combined_radial_velocity_model(np.concatenate((t_model1, t_model2)), *popt)[:len(t_model1)]
    vr_model2 = combined_radial_velocity_model(np.concatenate((t_model1, t_model2)), *popt)[len(t_model1):]
    
    # 잔차 계산은 원래 데이터 포인트 기준으로
    residuals1 = vr_observed1 - combined_radial_velocity_model(np.concatenate((t_data1, t_data2)), *popt)[:len(t_data1)]
    residuals2 = vr_observed2 - combined_radial_velocity_model(np.concatenate((t_data1, t_data2)), *popt)[len(t_data1):]
    rms = np.sqrt(np.mean(np.concatenate((residuals1, residuals2))**2))
    st.write(f"### 총 RMS 잔차 = {rms:.2f} km/s")
    
    # 그래프 그리기
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    
    # 데이터 세트 1 그래프
    axs.errorbar(t_data1, vr_observed1, yerr=1.5, fmt='ro', label='data set 1 (관측값)')
    axs.plot(t_model1, vr_model1, 'r-', label='data set 1 (fitted)')
    
    # 데이터 세트 2 그래프
    axs.errorbar(t_data2, vr_observed2, yerr=1.5, fmt='bo', label='data set 2 (관측값)')
    axs.plot(t_model2, vr_model2, 'b-', label='data set 2 (fitted)')
    
    axs.set_title('RV curve', fontproperties=font_prop)
    axs.set_xlabel('time', fontproperties=font_prop)
    axs.set_ylabel('RV (km/s)', fontproperties=font_prop)
    axs.legend(prop=font_prop)
    axs.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
 
 
if __name__ == '__main__':
    st.title("시선속도 데이터 피팅")
    
    st.markdown("### 데이터 세트 1 입력")
    t_data1_input = st.text_area("시간 데이터 (콤마로 구분)", value="", key="t_data1")
    vr_observed1_input = st.text_area("시선속도 데이터 (콤마로 구분)", value="", key="vr_data1")

    st.markdown("---")
    st.markdown("### 데이터 세트 2 입력")
    t_data2_input = st.text_area("시간 데이터 (콤마로 구분)", value="t_data1_input", key="t_data2")
    vr_observed2_input = st.text_area("시선속도 데이터 (콤마로 구분)", value="", key="vr_data2")
    
    try:
        t_data1 = np.array([float(x) for x in t_data1_input.split(",")])
        vr_observed1 = np.array([float(x) for x in vr_observed1_input.split(",")])
        t_data2 = np.array([float(x) for x in t_data2_input.split(",")])
        vr_observed2 = np.array([float(x) for x in vr_observed2_input.split(",")])
    except Exception:
        st.error("입력 데이터는 콤마로 구분된 숫자여야 합니다.")
        st.stop()
        
    if st.button("피팅 실행"):
        fit_and_plot(t_data1, vr_observed1, t_data2, vr_observed2)
