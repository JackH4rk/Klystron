import numpy as np
import scipy.constants as C
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.special import j1, jn_zeros
from scipy.interpolate import interp1d

# 电子运动方程参数
f0 = 9.3e9  # 电场频率 (Hz)
v_0 = 2.85e7  # 平均速度 (m/s)
delta_v = 1e7  # 速度振幅 (m/s)
T_0 = 1 / f0  # 射频周期 (s)
omega_0 = 2 * np.pi * f0  # 角频率

# 空间电荷场参数
Q = 1e-9  # 单个电子圆盘带电量
a = 5e-3
b = 4e-3
N_zeros = 70

# 腔场参数
R_Q = 128  # 腔的R/Q
Q_t = 298  # 品质因子
omega_cav = 2 * np.pi * 3.45e9  # 腔的谐振频率 (Hz)
v_e = 0.5 * C.c  # 电子速度 (m/s)
M = 0.688  # 耦合系数


# 电场空间形状函数 f(z)
def f(z, k=0.5 * v_e / np.sqrt(-np.log(M))):
    return (k / np.sqrt(np.pi)) * np.exp(-k ** 2 * (z) ** 2)


# 电子圆盘的速度 (时间变化)
def v_i(t, initial_speed=v_0, delta_v=delta_v):
    return initial_speed + delta_v * np.cos(omega_0 * t)


# 电子运动方程中的洛伦兹力计算
def lorentz(t, y, z_i, Q, a, b):
    z, pz = y[0::2], y[1::2]
    gamma_factor = np.sqrt(pz ** 2 + (C.m_e * C.c) ** 2)
    dz_dt = pz * C.c / gamma_factor  # dz/dt
    Ez = get_E(t, z, z_i, Q, a, b)
    dpz_dt = -C.e * Ez  # dpz/dt (Lorentz force)
    return np.stack([dz_dt, dpz_dt], axis=1).flatten()


# 电场函数 (空间电荷场)
def Es(z, Q, a, b):
    mu_0_ps = jn_zeros(0, N_zeros)
    arr = np.exp(-mu_0_ps * np.abs(z) / a) * (2 / mu_0_ps * j1(mu_0_ps * b / a) / j1(mu_0_ps)) ** 2 * np.sign(z)
    return Q / (2 * np.pi * C.epsilon_0 * b ** 2) * arr.sum()


# 总的空间电荷场 E_spch
def E_spch(z, z_i, Q, a, b):
    E_spch_total = np.zeros_like(z)
    for z_pos in z_i:
        E_spch_total += Es(z - z_pos, Q, a, b)
    return E_spch_total


# 腔场 E_cav 计算
def E_cav(t, z, z_i, Q_i=1e-9):
    velocities = v_i(t)  # 更新速度
    I_ind_t = np.sum(Q_i * velocities * f(z_i))  # 感应电流
    Z_cav = R_Q / (1 / Q_t + 1j * ((omega_0 ** 2 - omega_cav ** 2) / (omega_0 * omega_cav)))  # 腔阻抗
    abs_Z_cav = np.real(Z_cav)
    Vtg = np.abs(I_ind_t) * abs_Z_cav  # 腔压
    return Vtg * f(z)  # 腔场 E_cav


# 计算加速电场 Ez(t, z)
def get_E(t, z, z_i, Q, a, b):
    E_spch_vals = E_spch(z, z_i, Q, a, b)
    E_cav_vals = E_cav(t, z, z_i)
    return E_spch_vals + E_cav_vals  # 加速电场为空间电荷场与腔场之和


# 进行射频周期迭代计算，并根据收敛条件更新电场
def run_simulation(z_min, z_max, t_steps, z_steps, Q, a, b, tolerance=1e-6, max_cycles=100):
    # 时间点
    t = np.linspace(0, T_0, t_steps)

    # 初始条件
    initial_conditions = np.array([[z0, 5.4e-23] for z0 in np.linspace(z_min, z_max, z_steps)])

    # 存储每次计算的腔压，用于收敛判断
    Vtg_prev = None
    converged = False
    cycle = 0

    while not converged and cycle < max_cycles:
        # 使用 scipy.integrate 解算运动方程
        sol = solve_ivp(
            lorentz, t_span=(t[0], t[-1]), y0=initial_conditions.flatten(),
            method='RK45', t_eval=t, args=(initial_conditions[:, 0], Q, a, b)  # 传入 z_i, Q, a, b
        )

        # 计算空间电荷场 E_spch
        z_i = sol.y[::2, :]  # 电子圆盘位置
        E_spch_values = E_spch(np.linspace(z_min, z_max, 1000), z_i, Q, a, b)

        # 计算腔场 E_cav
        E_cav_values = E_cav(t[-1], np.linspace(z_min, z_max, 1000), z_i)  # 在最后一个时刻计算腔场

        # 线性插值，确保电场长度一致
        interp_E_spch = interp1d(np.linspace(z_min, z_max, 1000), E_spch_values, kind='linear',
                                 fill_value='extrapolate')
        interp_E_cav = interp1d(np.linspace(z_min, z_max, 1000), E_cav_values, kind='linear', fill_value='extrapolate')

        E_spch_values_interp = interp_E_spch(np.linspace(z_min, z_max, 500))
        E_cav_values_interp = interp_E_cav(np.linspace(z_min, z_max, 500))

        # 计算加速电场 Ez(t, z)
        Ez_values = E_spch_values_interp + E_cav_values_interp

        # 计算腔压 Vtg
        Vtg = np.sum(E_cav_values_interp)  # 假设腔压是腔场的总和，实际中可能需修改

        # 判断收敛：检查腔压的变化是否小于容差
        if Vtg_prev is not None:
            Vtg_diff = np.abs(Vtg - Vtg_prev) / Vtg_prev
            if Vtg_diff < tolerance:
                converged = True

        Vtg_prev = Vtg
        cycle += 1

    # 输出最终结果
    print(f"Simulation converged after {cycle} RF cycles.")
    return Ez_values


# 主程序执行
z_min, z_max = -0.01, 0.01  # 位置范围 (m)
t_steps = 500  # 时间步数
z_steps = 70  # 位置步数
Ez = run_simulation(z_min, z_max, t_steps, z_steps, Q, a, b)

# 绘制结果
plt.plot(np.linspace(z_min, z_max, 500), Ez)
plt.xlabel("Position (m)")
plt.ylabel("Electric Field (V/m)")
plt.title("Electric Field Distribution after Convergence")
plt.show()