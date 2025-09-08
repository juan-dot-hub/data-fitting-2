import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
# font_prop 변수 정의 (DejaVu Sans 기준)
font_prop = fm.FontProperties(family='DejaVu Sans')

def kepler_equation(E, M, e):
    return E - e * np.sin(E) - M

def solve_kepler(M, e):
    E_guess = M
    E = fsolve(kepler_equation, E_guess, args=(M, e))[0]
    return E

def mean_anomaly(t, P, t0):
    return 2 * np.pi * (t - t0) / P

def true_anomaly(E, e):
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

def radial_velocity_model(t, gamma, K, P, e, omega, t0):
    vr = np.zeros_like(t)
    for i, time in enumerate(t):
        M = mean_anomaly(time, P, t0)
        E = solve_kepler(M, e)
        theta = true_anomaly(E, e)
        vr[i] = gamma + K * (np.cos(theta + omega) + e * np.cos(omega))
    return vr

def fit_and_plot(t_data, vr_observed, label_prefix):
    # 데이터셋에 따라 초기 오메가 값 설정
    if label_prefix == "data set 1":
        initial_omega = 0.5 * np.pi
    elif label_prefix == "data set 2":
        initial_omega = 1.5 * np.pi
    else:
        initial_omega = 0.5 * np.pi # 기본값

    initial_guess = [
        np.mean(vr_observed),
        (np.max(vr_observed) - np.min(vr_observed)) / 2,
        40,
        0.4,
        initial_omega,
        0.0
    ]
    bounds = (
        [-100, 0, 1, 0, 0, -50],
        [100, 100, 200, 0.99, 2 * np.pi, 50]
    )
    popt, pcov = curve_fit(
        radial_velocity_model, t_data, vr_observed,
        p0=initial_guess, bounds=bounds, maxfev=5000
    )
    fitted_gamma, fitted_K, fitted_P, fitted_e, fitted_omega, fitted_t0 = popt
    
    # 모델 곡선을 부드럽게 그리기 위해 더 촘촘한 시간 배열 생성
    t_model = np.linspace(min(t_data), max(t_data), 500)
    vr_model = radial_velocity_model(t_model, *popt)
    
    # 잔차 계산은 원래 데이터 포인트 기준으로
    residuals = vr_observed - radial_velocity_model(t_data, *popt)
    rms = np.sqrt(np.mean(residuals ** 2))
    
    st.write(f"### {label_prefix} 시선속도 피팅 결과")
    st.write(f"γ = {fitted_gamma:.2f} ± {np.sqrt(pcov[0, 0]):.2f} km/s")
    st.write(f"K = {fitted_K:.2f} ± {np.sqrt(pcov[1, 1]):.2f} km/s")
    st.write(f"P = {fitted_P:.2f} ± {np.sqrt(pcov[2, 2]):.2f} days")
    st.write(f"e = {fitted_e:.3f} ± {np.sqrt(pcov[3, 3]):.3f}")
    st.write(f"ω = {fitted_omega:.3f} ± {np.sqrt(pcov[4, 4]):.3f} rad")
    st.write(f"t₀ = {fitted_t0:.2f} ± {np.sqrt(pcov[5, 5]):.2f} days")
    st.write(f"RMS 잔차 = {rms:.2f} km/s")
    
    return t_data, vr_observed, t_model, vr_model, label_prefix

if __name__ == '__main__':
    st.title("시선속도 데이터 피팅")
    st.markdown("### 데이터 세트 1 입력")
    t_data1_input = st.text_area("시간 데이터 (콤마로 구분)", 
                                 value="0.0, 1.32, 2.63, 3.95, 5.26, 6.58, 7.89, 9.21, 10.53, 11.84, 13.16, 14.47, 15.79, 17.11, 18.42, 19.74, 21.05, 22.37, 23.68, 25.0",
                                 key="t_data1")
    vr_observed1_input = st.text_area("시선속도 데이터 (콤마로 구분)", 
                                     value="25.22, 7.97, 0.89, 0.87, 0.05, 3.67, 11.55, 17.17, 23.62, 29.54, 15.46, 2.13, -0.94, -3.65, -0.57, 5.56, 10.86, 20.56, 26.51, 22.35",
                                     key="vr_data1")

    st.markdown("---")
    st.markdown("### 데이터 세트 2 입력 (직접 입력)")
    t_data2_input = st.text_area("시간 데이터 (콤마로 구분)", 
                                 value="",
                                 key="t_data2")
    vr_observed2_input = st.text_area("시선속도 데이터 (콤마로 구분)", 
                                     value="",
                                     key="vr_data2")

    try:
        t_data1 = np.array([float(x) for x in t_data1_input.split(",")])
        vr_observed1 = np.array([float(x) for x in vr_observed1_input.split(",")])
        t_data2 = np.array([float(x) for x in t_data2_input.split(",")])
        vr_observed2 = np.array([float(x) for x in vr_observed2_input.split(",")])
    except Exception:
        st.error("입력 데이터는 콤마로 구분된 숫자여야 합니다.")
        st.stop()

    if st.button("피팅 실행"):
        results1 = fit_and_plot(t_data1, vr_observed1, "data set 1")
        
        # 그래프 그리기
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        
        # Dataset 1 그래프
        axs.errorbar(results1[0], results1[1], yerr=1.5, fmt='ro', label=f'{results1[4]} observed')
        axs.plot(results1[2], results1[3], 'r-', label=f'{results1[4]} fitted curve')
        
        # Dataset 2가 비어 있지 않을 경우에만 그래프에 추가
        if len(vr_observed2) > 0 and len(t_data2) > 0:
            results2 = fit_and_plot(t_data2, vr_observed2, "data set 2")
            axs.errorbar(results2[0], results2[1], yerr=1.5, fmt='bo', label=f'{results2[4]} observed')
            axs.plot(results2[2], results2[3], 'b-', label=f'{results2[4]} fitted curve')
        
        axs.set_title('RV Curve (data set 1 & 2)', fontproperties=font_prop)
        axs.set_xlabel('Time (days)', fontproperties=font_prop)
        axs.set_ylabel('RV (km/s)', fontproperties=font_prop)
        axs.legend(prop=font_prop)
        axs.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
