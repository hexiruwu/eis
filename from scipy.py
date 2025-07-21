from scipy.optimize import curve_fit, minimize
import openpyxl
#from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from matplotlib.font_manager import FontProperties
chinese_font_path = '/System/Library/Fonts/Hiragino Sans GB.ttc'
chinese_font = FontProperties(fname=chinese_font_path)
# 数据文件路径和电池类型
data_path = pathlib.Path("Impedance raw data and fitting data")
battery_types = {
    'NCA': 'NCA battery',
    'NCM': 'NCM battery',
    'NCM+NCA': 'NCM+NCA battery'
}

def load_impedance_data(battery_type, file_name, sheet_name=None):
    file_path = pathlib.Path("Downloads/eis-main") / data_path / battery_types[battery_type] / file_name
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return None
    if sheet_name is None:
        xl_file = pd.ExcelFile(file_path, engine="openpyxl")
        return xl_file.sheet_names
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        df.columns = df.columns.map(str)
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return None



def cpe_impedance(Q, n, w):
    w = np.where(w == 0, 1e-12, w)  # 防止w为0
    return 1 / (Q * (1j * w) ** n)

def warburg_impedance(W, w):
    w = np.where(w == 0, 1e-12, w)
    return W / np.sqrt(1j * w)

def circuit_impedance(f, R0, R1, Q1, n1, R2, Q2, n2, W, L=0):
    w = 2 * np.pi * f
    Z_L = 1j * w * L
    # 第一并联支路
    Z_cpe1 = cpe_impedance(Q1, n1, w)
    Z_R1_cpe1 = 1 / (1/R1 + 1/Z_cpe1)
    # 第二并联支路
    Z_warburg = warburg_impedance(W, w)
    Z_R2_w = R2 + Z_warburg
    Z_cpe2 = cpe_impedance(Q2, n2, w)
    Z_R2w_cpe2 = 1 / (1/Z_R2_w + 1/Z_cpe2)
    # 总阻抗
    Z_total = Z_L + R0 + Z_R1_cpe1 + Z_R2w_cpe2
    return np.real(Z_total), np.imag(Z_total)

def fit_function(data, iterations=10000):
    # 提取频率和阻抗实部、虚部
    freq = data["Data: Frequency"].values
    z_real = data["Data: Z'"].values
    z_imag = data["Data: Z''"].values


    # 拟合目标函数，将实部和虚部拼接
    def target_func(f, R0, R1, Q1, n1, R2, Q2, n2, W, L):
        zr, zi = circuit_impedance(f, R0, R1, Q1, n1, R2, Q2, n2, W, L)
        return np.concatenate([zr, zi])

    # 拼接实部和虚部作为y
    ydata = np.concatenate([z_real, z_imag])

    p0 = [
        1,    # R0
        0.2,    # R1
        4,  # Q1
        0.8,   # n1
        0.5,    # R2
        4,  # Q2
        0.8,   # n2
        1,    # W
        1e-7,  # L 电感
    ]
    bounds = (
    [0, 0.0005, 0, 0, 0.0005, 0, 0, 0.001, 0],         # 所有参数下限
    [np.inf, 10, 10, 1, np.inf, 10, 1, 15, 1e-4]  # 上限，n1/n2最大为1
)

    # 拟合
    popt, pcov = curve_fit(target_func, freq, ydata, p0=p0, bounds=bounds, maxfev=iterations)

    param_names = ["R0", "R1", "Q1", "n1", "R2", "Q2", "n2", "W", "L"]
    fit_dict = dict(zip(param_names, popt))
    return fit_dict
def plot_fit_comparison(data, my_params):
    freq = data["Data: Frequency"].values
    z_real = data["Data: Z'"].values
    z_imag = data["Data: Z''"].values
# 你的拟合
    my_zr, my_zi = circuit_impedance(freq, *my_params.values())

    plt.figure(figsize=(8, 6))
    # 原始数据
    if "Data: Z'" in data.columns and "Data: Z''" in data.columns:
        plt.scatter(z_real, -z_imag, alpha=0.7, s=30, label="原始数据", color="black")
    # 别人的拟合数据
    
    #if "Fit: Z'" in data.columns and "Fit: Z''" in data.columns:
    #    plt.plot(data["Fit: Z'"].values, -data["Fit: Z''"].values, 'r-', linewidth=2, label="他人拟合", alpha=0.8)
    plt.tight_layout()
    plt.plot(my_zr, -my_zi, label="我的拟合", color="blue")
    plt.xlabel("Z' (Ω)")
    plt.ylabel("-Z'' (Ω)")
    plt.legend(prop=chinese_font)
    plt.title("Nyquist 拟合对比", fontproperties=chinese_font)
    plt.axis('equal')
    plt.show()
# 示例：加载NCM电池的第一个工作表并拟合
battery_type = 'NCM'
file_name = 'CY25_0.5_1.xlsx'
sheets = load_impedance_data(battery_type, file_name)
# if sheets:
#     data = load_impedance_data(battery_type, file_name, sheets[1])
#     if data is not None:
#         fit_result = fit_function(data)
#         plot_fit_comparison(data, fit_result)
#         print("拟合参数--format：", fit_result)
all_fit_results = []
if sheets:
    for sheet in sheets:
        print(f"\n正在处理工作表: {sheet}")
        data = load_impedance_data(battery_type, file_name, sheet)
        if data is not None:
            try:
                fit_result = fit_function(data)
                plot_fit_comparison(data, fit_result)

                # 将结果加入列表（并加上 sheet 名）
                result_with_sheet = {'sheet': sheet}
                result_with_sheet.update(fit_result)
                all_fit_results.append(result_with_sheet)

                print("拟合参数:", fit_result)
            except Exception as e:
                print(f"拟合失败: {e}")
        else:
            print("数据加载失败")

    # 保存所有结果到 Excel
    result_df = pd.DataFrame(all_fit_results)
    output_path = pathlib.Path("Downloads/eis-main") / f"{battery_type}{file_name}_fitting_results.xlsx"
    result_df.to_excel(output_path, index=False)
    print(f"\n所有拟合参数已保存到: {output_path}")