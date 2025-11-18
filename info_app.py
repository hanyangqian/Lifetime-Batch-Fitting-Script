import sys
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QTableWidgetItem, 
                            QMessageBox, QGraphicsScene, QApplication, 
                            QHeaderView, QFileDialog, QTreeWidgetItem)
from PyQt5 import QtWidgets, QtGui
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import QTimer, QCoreApplication, pyqtSignal, QObject, QDir, QSettings
from PyQt5.QtGui import QImage, QPixmap, QFont, QCursor
from Ui_info_window import Ui_MainWindow as main_window
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pickle
import numpy as np
import zipfile
import tempfile
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
import copy
from numpy import array, float64, arange
import traceback
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.stats import chi2
import warnings
from enum import Enum
import inspect

# 常量定义
DEFAULT_FIT_OPTIONS = {
    'DiffMinChange': 1e-8,
    'DiffMaxChange': 0.1,
    'MaxIter': 400,
    'MaxFunEvals': 600,
    'TolFun': 1e-6,
    'TolX': 1e-6
}

UNIT_MAPPING = {'s': 1, 'ms': 1000, 'μs': 1000000, 'ns': 1000000000}

class RobustMethod(Enum):
    OFF = 'Off'
    LAR = 'LAR'
    BISQUARE = 'Bisquare'

class Algorithm(Enum):
    TRUST_REGION = 'Trust-Region'
    LEVENBERG_MARQUARDT = 'Levenberg-Marquardt'

class MATLABCurveFitOptions:
    """完全对应MATLAB的fitoptions参数"""
    def __init__(self):
        # 稳健性选项 (MATLAB: 'Robust')
        self.Robust = RobustMethod.OFF
        
        # 算法选项 (MATLAB: 'Algorithm')
        self.Algorithm = Algorithm.LEVENBERG_MARQUARDT
        
        # 差分步长设置 (MATLAB: 'DiffMinChange', 'DiffMaxChange')
        self.DiffMinChange = 1e-8
        self.DiffMaxChange = 0.1
        
        # 迭代控制 (MATLAB: 'MaxIter', 'MaxFunEvals')
        self.MaxIter = 400
        self.MaxFunEvals = 600
        
        # 收敛容差 (MATLAB: 'TolFun', 'TolX')
        self.TolFun = 1e-6
        self.TolX = 1e-6
        
        # 参数边界 (MATLAB: 'Lower', 'Upper')
        self.Lower = -np.inf
        self.Upper = np.inf
        
        # 起始点 (MATLAB: 'StartPoint')
        self.StartPoint = None
        
        # 权重 (MATLAB: 'Weights')
        self.Weights = None
        
        # 排除点 (MATLAB: 'Exclude')
        self.Exclude = None

class MATLABFitResult:
    """完全对应MATLAB的fitresult输出"""
    def __init__(self):
        # 基本拟合结果
        self.success = False
        self.message = ""
        self.iterations = 0
        self.funcCount = 0
        self.algorithm = ""
        
        # 参数估计
        self.params = None
        self.param_names = None
        
        # 拟合统计
        self.sse = 0.0  # 误差平方和
        self.rsquare = 0.0  # R²
        self.dfe = 0  # 误差自由度
        self.adjrsquare = 0.0  # 调整R²
        self.rmse = 0.0  # 均方根误差
        
        # 残差分析
        self.residuals = None
        
        # 雅可比矩阵 (用于误差分析)
        self.jacobian = None
        
    def __str__(self):
        """类似MATLAB的输出格式"""
        output = []
        output.append("拟合结果:")
        output.append(f"     算法: {self.algorithm}")
        output.append(f"     收敛: {'是' if self.success else '否'}")
        output.append(f"     迭代次数: {self.iterations}")
        output.append(f"     函数计算次数: {self.funcCount}")
        output.append("")
        output.append("系数估计 (95% 置信区间):")
        
        for i, (name, value) in enumerate(zip(self.param_names, self.params)):
            output.append(f"     {name} = {value:.6f}")
        
        output.append("")
        output.append("拟合优度统计:")
        output.append(f"     SSE: {self.sse:.6f}")
        output.append(f"     R²: {self.rsquare:.6f}")
        output.append(f"     调整R²: {self.adjrsquare:.6f}")
        output.append(f"     RMSE: {self.rmse:.6f}")
        
        return "\n".join(output)

class MATLABCurveFitter:
    """完全复现MATLAB的fit函数功能"""
    
    def __init__(self):
        self.options = MATLABCurveFitOptions()
    
    def set_options(self, **kwargs):
        """设置拟合选项，参数名与MATLAB一致"""
        valid_options = ['Robust', 'Algorithm', 'DiffMinChange', 'DiffMaxChange', 
                        'MaxIter', 'MaxFunEvals', 'TolFun', 'TolX', 'Lower', 'Upper', 
                        'StartPoint', 'Weights', 'Exclude']
        
        for key, value in kwargs.items():
            if hasattr(self.options, key):
                setattr(self.options, key, value)
            else:
                warnings.warn(f"MATLAB中不支持的选项: {key}")
    
    def _robust_weight_function(self, residuals, method):
        """MATLAB稳健权重函数"""
        if method == RobustMethod.OFF:
            return np.ones_like(residuals)
        
        # 计算标准化残差
        mad = np.median(np.abs(residuals - np.median(residuals)))
        if mad == 0:
            mad = np.mean(np.abs(residuals))
        u = residuals / (1.4826 * mad)  # 一致性常数
        
        if method == RobustMethod.LAR:
            # LAR (最小绝对残差) - MATLAB实现
            weights = np.ones_like(u)
            mask = np.abs(u) > 1e-8
            weights[mask] = 1.0 / np.abs(u[mask])
            
        elif method == RobustMethod.BISQUARE:
            # Bisquare (Tukey's biweight) - MATLAB实现
            weights = np.zeros_like(u)
            mask = np.abs(u) < 4.685  # MATLAB默认调谐常数
            weights[mask] = (1 - (u[mask]/4.685)**2)**2
        
        return weights
    
    def _apply_exclude(self, x, y, exclude):
        """应用排除点"""
        if exclude is None:
            return x, y, np.ones_like(x, dtype=bool)
        
        if callable(exclude):
            include_mask = ~exclude(x, y)
        else:
            include_mask = ~exclude
            
        return x[include_mask], y[include_mask], include_mask
    
    def _apply_weights(self, residuals, weights):
        """应用权重"""
        if weights is None:
            return residuals
        return residuals * np.sqrt(weights)
    
    def _calculate_goodness_of_fit(self, y, y_pred, residuals, n_params, weights=None):
        """计算拟合优度统计量 (MATLAB方法)"""
        n = len(y)
        
        # 加权残差
        if weights is not None:
            weighted_residuals = residuals * np.sqrt(weights)
            ss_res = np.sum(weighted_residuals**2)
        else:
            ss_res = np.sum(residuals**2)
        
        # 加权总平方和
        if weights is not None:
            y_weighted = y * np.sqrt(weights)
            y_mean_weighted = np.average(y, weights=weights)
            ss_tot = np.sum((y_weighted - y_mean_weighted)**2)
        else:
            ss_tot = np.sum((y - np.mean(y))**2)
        
        # 计算统计量
        sse = ss_res
        rsquare = 1 - ss_res/ss_tot if ss_tot != 0 else 0
        dfe = n - n_params  # 误差自由度
        adjrsquare = 1 - (1 - rsquare) * (n - 1) / dfe if dfe > 0 else rsquare
        rmse = np.sqrt(sse / n)
        
        return sse, rsquare, dfe, adjrsquare, rmse
    
    def fit_curve(self, x, y, model_type, initial_guess=None, param_names=None):
        """
        拟合预定义模型 (完全复现MATLAB fit函数)
        
        参数:
            x: 自变量数据
            y: 因变量数据
            model_type: 模型类型 ('linear', 'exponential', 'polynomial2', 'polynomial3', 
                                'power', 'logarithmic', 'gaussian', 'sine')
            initial_guess: 参数初始值
            param_names: 参数名称
            
        返回:
            MATLABFitResult对象，包含所有MATLAB输出
        """
        
        # 获取模型定义
        model_def = self._get_model_definition(model_type)
        if model_def is None:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        func = model_def['function']
        default_param_names = model_def['param_names']
        n_params = len(default_param_names)

        # 设置参数名称
        if param_names is None:
            param_names = default_param_names
        
        # 设置初始猜测
        if initial_guess is None:
            if self.options.StartPoint is not None:
                initial_guess = self.options.StartPoint
            else:
                initial_guess = self._get_default_start_point(model_type, x, y)
        
        # 应用排除点
        x_fit, y_fit, include_mask = self._apply_exclude(x, y, self.options.Exclude)
        
        # 执行拟合
        result = self._fit_implementation(x_fit, y_fit, func, initial_guess, param_names)
        
        # 计算完整数据的残差
        y_pred_full = func(x, *result.params)
        result.residuals = y - y_pred_full
        
        return result
    
    
    def _get_model_definition(self, model_type):
        """修改双指数模型定义"""
        models = {
            'linear': {
                'function': lambda x, a, b: a * x + b,
                'param_names': ['a', 'b'],
                'description': 'y = a*x + b'
            },
            '单指数': {
                'function': lambda x, a, b, c: a * np.exp(-x/b) + c,  # 修改为 -x/b
                'param_names': ['a', 'b', 'c'],
                'description': 'y = a*exp(-x/b) + c'
            },
            '双指数': {
                'function': lambda x, a, t1, c, t2, e: a * np.exp(-x/t1) + c * np.exp(-x/t2) + e,
                'param_names': ['a', 't1', 'c', 't2', 'e'],
                'description': 'y = a*exp(-x/t1) + c*exp(-x/t2) + e'
            },
            'polynomial2': {
                'function': lambda x, a, b, c: a * x**2 + b * x + c,
                'param_names': ['a', 'b', 'c'],
                'description': 'y = a*x² + b*x + c'
            },
        }
        return models.get(model_type)

    def _get_default_start_point(self, model_type, x, y):
        """修改双指数模型的初始参数计算"""
        if model_type == '双指数':
            y_range = np.max(y) - np.min(y)
            if y_range == 0:
                y_range = 1.0
                
            # 计算初始时间常数估计 (t1和t2)
            # 方法1：使用数据的分位数估计衰减时间
            decay_10_idx = np.argmin(np.abs(y - (np.max(y)*0.1)))
            decay_50_idx = np.argmin(np.abs(y - (np.max(y)*0.5)))
            
            # 确保有足够的点用于估计
            if decay_50_idx < 1:
                decay_50_idx = len(x) // 2
            if decay_10_idx <= decay_50_idx:
                decay_10_idx = decay_50_idx + len(x) // 4
                
            # 计算时间常数 (t1和t2)，确保 t1 < t2
            t1_guess = x[decay_50_idx] / np.log(2)  # 快速衰减分量（半衰期估计）
            t2_guess = x[decay_10_idx] / np.log(10)  # 慢速衰减分量（1/10衰减期估计）
            
            # 确保 t1 < t2（快速分量衰减更快）
            if t1_guess >= t2_guess:
                t1_guess, t2_guess = t2_guess, t1_guess
                # 如果交换后仍然不满足，设置默认比例
                if t1_guess >= t2_guess:
                    t1_guess = t2_guess / 5
            
            # 确保时间常数为正值
            t1_guess = max(t1_guess, 1e-6)
            t2_guess = max(t2_guess, t1_guess * 1.1)
            
            return [
                y_range * 0.7,  # a (快速分量振幅)
                t1_guess,       # t1 (快速衰减时间常数)
                y_range * 0.3,  # c (慢速分量振幅)
                t2_guess,       # t2 (慢速衰减时间常数)
                np.min(y)       # e (基线)
            ]
        elif model_type == '单指数':
            # 单指数衰减模型的初始值计算
            y_range = np.max(y) - np.min(y)
            if y_range == 0:
                y_range = 1.0
            
            # 使用对数线性回归估计衰减率
            y_adj = y - np.min(y) + 1e-10
            valid_idx = y_adj > 0.1 * np.max(y_adj)
            if np.sum(valid_idx) > 2:
                log_y = np.log(y_adj[valid_idx])
                slope, _ = np.polyfit(x[valid_idx], log_y, 1)
                tau = -1.0/slope if slope < 0 else -0.1
            else:
                tau = -0.1
                
            return [y_range, 1.0/tau, np.min(y)]
        
        # 其他模型的默认初始值
        return [1.0] * len(self._get_model_definition(model_type)['param_names'])
        
        
    def _fit_implementation(self, x, y, func, initial_guess, param_names):
        """改进的拟合实现核心"""
        result = MATLABFitResult()
        result.param_names = param_names
        
        # 设置优化选项
        lsq_options = {
            'method': 'trf' if self.options.Algorithm == Algorithm.TRUST_REGION else 'lm',
            'max_nfev': self.options.MaxFunEvals,
            'ftol': self.options.TolFun,
            'xtol': self.options.TolX,
            'gtol': self.options.TolX,
            'x_scale': 'jac',
            'verbose': 2,  # 输出详细优化信息
            'tr_solver': 'lsmr',  # 使用更稳定的求解器
            'loss': 'soft_l1' if self.options.Robust != RobustMethod.OFF else 'linear',
        }
        
        # 设置参数边界 (特别是对指数衰减模型)
        if 'exp' in func.__code__.co_name.lower():  # 如果是指数模型
            bounds_lower = np.full_like(initial_guess, -np.inf)
            bounds_upper = np.full_like(initial_guess, np.inf)
            
            # 对衰减率参数设置合理边界
            for i, name in enumerate(param_names):
                if name in ['t1', 't2', 'b', 'd']:  # 衰减率参数
                    bounds_lower[i] = 1e-6  # 最小衰减时间常数
                    bounds_upper[i] = 1000  # 最大衰减时间常数
                elif name in ['a', 'c']:  # 振幅参数
                    bounds_lower[i] = 0  # 振幅必须非负
            
            # 如果是双指数模型，添加 t1 < t2 的约束
            if len(param_names) >= 5 and 't1' in param_names and 't2' in param_names:
                t1_idx = param_names.index('t1')
                t2_idx = param_names.index('t2')
                # 确保 t1 < t2 的边界约束
                bounds_upper[t1_idx] = bounds_upper[t2_idx]  # t1 的上限不超过 t2 的上限
            
            lsq_options['bounds'] = (bounds_lower, bounds_upper)
        
        try:
            # 定义目标函数
            def objective(params):
                # 如果是双指数模型，强制 t1 < t2
                if len(params) >= 5 and hasattr(self, 'current_model_type') and self.current_model_type == '双指数':
                    t1_idx = 1  # t1 在参数列表中的位置
                    t2_idx = 3  # t2 在参数列表中的位置
                    if params[t1_idx] >= params[t2_idx]:
                        # 如果 t1 >= t2，返回很大的残差来惩罚这种配置
                        return np.full_like(y, 1e10)
                
                y_pred = func(x, *params)
                residuals = y - y_pred
                
                # 应用稳健权重
                if self.options.Robust != RobustMethod.OFF:
                    weights = self._robust_weight_function(residuals, self.options.Robust)
                    residuals = residuals * np.sqrt(weights)
                
                # 应用用户权重
                if self.options.Weights is not None:
                    residuals = residuals * np.sqrt(self.options.Weights)
                
                return residuals
            
            # 执行拟合 - 分阶段优化策略
            print(f"初始参数: {initial_guess}")
            
            # 第一阶段: 宽松容差快速拟合
            phase1_options = lsq_options.copy()
            phase1_options['ftol'] = 1e-6
            phase1_options['xtol'] = 1e-6
            phase1_options['max_nfev'] = min(200, self.options.MaxFunEvals//2)
            
            phase1_result = least_squares(objective, initial_guess, **phase1_options)
            print(f"第一阶段结果: {phase1_result.x}")
            
            # 第二阶段: 严格容差精细拟合
            phase2_options = lsq_options.copy()
            phase2_result = least_squares(objective, phase1_result.x, **phase2_options)
            print(f"第二阶段结果: {phase2_result.x}")
            
            # 保存最终结果
            result.params = phase2_result.x
            result.success = phase2_result.success
            result.message = phase2_result.message
            result.iterations = phase2_result.nfev
            result.funcCount = phase2_result.nfev
            result.jacobian = phase2_result.jac
            
            # 计算拟合优度统计量
            y_pred = func(x, *result.params)
            residuals = y - y_pred
            
            sse, rsquare, dfe, adjrsquare, rmse = self._calculate_goodness_of_fit(
                y, y_pred, residuals, len(param_names))
            
            result.sse = sse
            result.rsquare = rsquare
            result.dfe = dfe
            result.adjrsquare = adjrsquare
            result.rmse = rmse
            result.residuals = residuals
            
        except Exception as e:
            result.success = False
            result.message = f"拟合失败: {str(e)}"
            print(f"拟合错误: {str(e)}")
            print(traceback.format_exc())
        
        return result

class mainWindow(QMainWindow, main_window):
    def __init__(self, parent=None, product_names=None):
        super(mainWindow, self).__init__(parent)
        self.setupUi(self)
        
        # 使用常量初始化
        self.temperature_calibration = [50,100,150,200,250,300,350,400,450,500,550,600]
        self.lifetime_calibration = [3.646040506,3.212977315,2.842532971,2.519596718,2.236712857,1.981563617,1.710492252,1.280798396,0.688913014,0.379551499,0.321295208,0.317258355]

        # 初始化设置
        self.settings = QSettings("YourCompany", "PMT_Analysis")
        self.setup_matplotlib_chinese()

        # 连接信号槽
        self._connect_signals()
        
        # 初始化变量
        self._initialize_variables()
        
        # 初始化UI组件
        self._initialize_ui()
        
        # 恢复设置
        self.restore_previous_settings()

    def _connect_signals(self):
        """集中连接所有信号槽"""
        # 现有连接...
        self.treeWidget.itemClicked.connect(self.calculate_decay)
        self.comboBox_10.currentIndexChanged.connect(self.comboBox_10_changed)
        self.comboBox.currentIndexChanged.connect(self.fit_curve_model)
        
        # 文件夹设置
        self.pushButton.clicked.connect(self.set_work_directory)
        self.pushButton_2.clicked.connect(self.select_target_files)
        self.pushButton_3.clicked.connect(self.set_download_directory)
        
        # 下载按钮
        self.pushButton_5.clicked.connect(self.save_all)
        self.pushButton_7.clicked.connect(self.save_one)
        
        # 数据筛选
        self.comboBox_3.currentTextChanged.connect(self.change_range)
        self.comboBox_4.currentTextChanged.connect(self.change_range)
        
        # 单位设置
        self.comboBox_5.currentTextChanged.connect(self.change_time_unit)
        self.comboBox_6.currentTextChanged.connect(self.change_time_unit)
        self.comboBox_7.currentTextChanged.connect(self.change_time_unit)
        
        # 寿命、温度计算
        self.pushButton_6.clicked.connect(self.calculate_decay)
        
        # 新增：导入标定数据
        self.pushButton_4.clicked.connect(self.import_calibration_data)
    
    def import_calibration_data(self):
        """导入标定数据 - CSV格式，包含'温度,寿命'列，并在新窗口显示图像"""
        try:
            # 获取初始目录
            initial_dir = self.pushButton.text() if self.pushButton.text() else QDir.homePath()
            
            # 打开文件选择对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "导入标定数据",
                initial_dir,
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if not file_path:
                return
                
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查必要的列
            required_columns = ['温度', '寿命']
            if not all(col in df.columns for col in required_columns):
                QMessageBox.warning(
                    self, 
                    "格式错误", 
                    f"CSV文件必须包含列: {', '.join(required_columns)}\n"
                    f"当前文件的列: {', '.join(df.columns.tolist())}"
                )
                return
            
            # 提取数据（单位保持为秒s）
            temperatures = df['温度'].values
            lifetimes = df['寿命'].values  # 单位：秒(s)
            
            # 验证数据
            if len(temperatures) != len(lifetimes):
                QMessageBox.warning(self, "数据错误", "温度和寿命数据长度不一致")
                return
                
            if len(temperatures) == 0:
                QMessageBox.warning(self, "数据错误", "没有找到有效数据")
                return
            
            # 创建标定数据预览窗口（单位保持为秒s）
            self.show_calibration_preview(temperatures, lifetimes, file_path)
            
        except Exception as e:
            QMessageBox.critical(self, "导入错误", f"导入标定数据时出错:\n{str(e)}")
            print(f"导入标定数据错误: {str(e)}")
            traceback.print_exc()
            
    def show_calibration_preview(self, temperatures, lifetimes, file_path):
        """显示标定数据预览窗口"""
        # 创建新窗口
        self.calibration_window = QtWidgets.QDialog(self)
        self.calibration_window.setWindowTitle("标定数据预览")
        self.calibration_window.setMinimumSize(800, 600)
        
        # 创建布局
        layout = QtWidgets.QVBoxLayout(self.calibration_window)
        
        # 添加文件信息标签
        file_info = QtWidgets.QLabel(f"文件: {os.path.basename(file_path)}")
        file_info.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(file_info)
        
        # 添加数据统计标签（使用科学计数法显示，单位：秒）
        stats_text = (f"数据点数: {len(temperatures)} | "
                    f"温度范围: {min(temperatures):.1f} - {max(temperatures):.1f} °C | "
                    f"寿命范围: {min(lifetimes):.5e} - {max(lifetimes):.5e} s")
        stats_label = QtWidgets.QLabel(stats_text)
        stats_label.setStyleSheet("font-size: 10pt;")
        layout.addWidget(stats_label)
        
        # 创建matplotlib图形
        figure = Figure(figsize=(8, 3))
        canvas = FC(figure)
        toolbar = NavigationToolbar(canvas, self.calibration_window)
        
        # 绘制标定曲线
        ax = figure.add_subplot(111)
        
        # 对数据进行排序以便连线
        sorted_indices = np.argsort(lifetimes)
        sorted_lifetimes = lifetimes[sorted_indices]
        sorted_temperatures = temperatures[sorted_indices]
        
        # 绘制散点图
        ax.scatter(lifetimes, temperatures, color='blue', s=50, alpha=0.7, label='标定数据点')
        
        # 单纯连线，不进行拟合
        ax.plot(sorted_lifetimes, sorted_temperatures, 'r-', linewidth=2, label='数据连线')
        
        ax.set_xlabel('寿命 (s)')  # 修改为单位：秒
        ax.set_ylabel('温度 (°C)')
        ax.set_title('温度-寿命标定曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置x轴为科学计数法格式
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1e'))
        
        figure.tight_layout()
        
        # 添加matplotlib组件到布局
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        # 添加数据表格
        table_label = QtWidgets.QLabel("标定数据表:")
        table_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(table_label)
        
        # 创建表格
        table = QtWidgets.QTableWidget()
        table.setRowCount(len(temperatures))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(['温度 (°C)', '寿命 (s)'])  # 修改为单位：秒
        
        # 填充表格数据（使用科学计数法）
        for i, (temp, life) in enumerate(zip(temperatures, lifetimes)):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{temp:.2f}"))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{life:.5e}"))
        
        table.setMaximumHeight(500)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(table)
        
        # 添加按钮区域
        button_layout = QtWidgets.QHBoxLayout()
        
        # 确认导入按钮
        confirm_button = QtWidgets.QPushButton("确认导入")
        confirm_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        confirm_button.clicked.connect(lambda: self.confirm_calibration_import(temperatures, lifetimes))
        
        # 取消按钮
        cancel_button = QtWidgets.QPushButton("取消")
        cancel_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }")
        cancel_button.clicked.connect(self.calibration_window.reject)
        
        button_layout.addWidget(confirm_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        # 显示窗口
        self.calibration_window.exec_()
    
    def confirm_calibration_import(self, temperatures, lifetimes):
        """确认导入标定数据"""
        try:
            # 更新标定数据（单位：秒s）
            self.temperature_calibration = temperatures.tolist()
            self.lifetime_calibration = lifetimes.tolist()
            
            # 关闭预览窗口
            if self.calibration_window:
                self.calibration_window.accept()
                self.calibration_window = None
            
            # 显示成功信息（使用科学计数法，单位：秒）
            QMessageBox.information(
                self, 
                "导入成功", 
                f"成功导入 {len(temperatures)} 个标定数据点\n"
                f"温度范围: {min(temperatures):.2f} - {max(temperatures):.2f} °C\n"
                f"寿命范围: {min(lifetimes):.5e} - {max(lifetimes):.5e} s"
            )
            
            # 打印导入信息（使用科学计数法，单位：秒）
            print(f"导入标定数据: {len(temperatures)} 个点")
            print(f"温度: {temperatures}")
            print(f"寿命 (s): [{min(lifetimes):.5e}, {max(lifetimes):.5e}]")
            
        except Exception as e:
            QMessageBox.critical(self, "导入错误", f"确认导入时出错:\n{str(e)}")
            print(f"确认导入标定数据错误: {str(e)}")

    def _initialize_variables(self):
        """初始化实例变量"""
        self.result = None
        self.file_path = None
        self.img_directory = None
        self.download_directory = None
        self.file_data_cache = {}
        self.lifetime = None
        self.temperature_measure = None
        self.current_model_type = None
        
        # 时间单位
        self.time_unit_5 = 1
        self.time_unit_6 = 1
        self.time_unit_7 = 1
        
        # 新增：标定数据相关变量
        self.temperature_calibration = []
        self.lifetime_calibration = []
        self.calibration_window = None

    def _initialize_ui(self):
        """初始化UI组件"""
        self.lineEdit_8.setEnabled(False)
        self.lineEdit_3.setEnabled(True)
        self.lineEdit_4.setEnabled(False)
        
        # 初始化treeWidget
        self.treeWidget.setHeaderLabels(["文件列表"])
        self.treeWidget.setColumnCount(1)
        
        # 初始化matplotlib图形
        self._setup_matplotlib_canvases()

    def _setup_matplotlib_canvases(self):
        """设置matplotlib画布 - 最小化工具栏版本"""
        # 第一个图形
        self.figure = Figure(figsize=(5, 4))  # 设置图形尺寸
        self.figure.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.92)
        self.canvas = FC(self.figure)
        self.toolbar = self._create_minimal_toolbar(self.canvas)
        self._add_canvas_to_layout(self.graphicsView, self.toolbar, self.canvas)
        
        # 第二个图形
        self.figure2 = Figure(figsize=(5, 4))
        self.figure2.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.92)
        self.canvas2 = FC(self.figure2)
        self.toolbar2 = self._create_minimal_toolbar(self.canvas2)
        self._add_canvas_to_layout(self.graphicsView_2, self.toolbar2, self.canvas2)
        
        # 第三个图形
        self.figure3 = Figure(figsize=(5, 4))
        self.figure3.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.92)
        self.canvas3 = FC(self.figure3)
        self.toolbar3 = self._create_minimal_toolbar(self.canvas3)
        self._add_canvas_to_layout(self.graphicsView_3, self.toolbar3, self.canvas3)

    def _create_minimal_toolbar(self, canvas):
        """创建最小化工具栏 - 只保留关键功能"""
        toolbar = NavigationToolbar(canvas, self)
        
        # 移除不必要的按钮
        actions_to_remove = []
        for action in toolbar.actions():
            text = action.text().lower() if action.text() else ""
            if any(word in text for word in ['subplots', 'customize', 'save']):
                actions_to_remove.append(action)
        
        for action in actions_to_remove:
            toolbar.removeAction(action)
        
        # 设置小尺寸样式
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #f0f0f0;
                spacing: 1px;
                padding: 0px;
                margin: 0px;
                border: none;
            }
            QToolButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 1px;
                margin: 0px;
            }
            QToolButton:hover {
                background-color: #d0d0d0;
                border: 1px solid #a0a0a0;
            }
            QToolButton:pressed {
                background-color: #b0b0b0;
            }
        """)
        
        # 设置固定小尺寸
        toolbar.setFixedHeight(22)
        
        # 设置小图标
        for action in toolbar.actions():
            widget = toolbar.widgetForAction(action)
            if widget and hasattr(widget, 'setIconSize'):
                widget.setIconSize(QtCore.QSize(14, 14))
                widget.setFixedSize(20, 18)
        
        return toolbar

    def _add_canvas_to_layout(self, graphics_view, toolbar, canvas):
        """将matplotlib组件添加到布局 - 优化空间利用"""
        # 清除现有布局
        if graphics_view.layout() is not None:
            QtWidgets.QWidget().setLayout(graphics_view.layout())
        
        layout = QtWidgets.QVBoxLayout(graphics_view)
        layout.setContentsMargins(1, 1, 1, 1)  # 最小边距
        layout.setSpacing(1)  # 最小间距
        
        if toolbar:
            layout.addWidget(toolbar)
        
        layout.addWidget(canvas)
        
        # 设置graphicsView的尺寸策略，优先给图形更多空间
        graphics_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def _safe_float_conversion(self, text, default=0.0):
        """安全的浮点数转换"""
        try:
            return float(text) if text else default
        except ValueError:
            return default

    def _safe_int_conversion(self, text, default=0):
        """安全的整数转换"""
        try:
            return int(text) if text else default
        except ValueError:
            return default

    def get_fit_options(self):
        """安全获取拟合选项"""
        return {
            'DiffMinChange': self._safe_float_conversion(self.lineEdit_19.text(), DEFAULT_FIT_OPTIONS['DiffMinChange']),
            'DiffMaxChange': self._safe_float_conversion(self.lineEdit_20.text(), DEFAULT_FIT_OPTIONS['DiffMaxChange']),
            'MaxIter': self._safe_int_conversion(self.lineEdit_21.text(), DEFAULT_FIT_OPTIONS['MaxIter']),
            'MaxFunEvals': self._safe_int_conversion(self.lineEdit_22.text(), DEFAULT_FIT_OPTIONS['MaxFunEvals']),
            'TolFun': self._safe_float_conversion(self.lineEdit_23.text(), DEFAULT_FIT_OPTIONS['TolFun']),
            'TolX': self._safe_float_conversion(self.lineEdit_24.text(), DEFAULT_FIT_OPTIONS['TolX'])
        }

    def _get_file_filters(self):
        """获取文件过滤器"""
        return (
            "Supported Files (*.xlsx *.xls *.txt *.csv);;"
            "Excel Files (*.xlsx *.xls);;"
            "Text Files (*.txt);;"
            "CSV Files (*.csv);;"
            "All Files (*.*)"
        )

    def _validate_file_extension(self, file_path):
        """验证文件扩展名"""
        return file_path.lower().endswith(('.xlsx', '.xls', '.txt', '.csv'))

    def _create_tree_item(self, file_path):
        """创建树形项目"""
        file_name = os.path.basename(file_path)
        item = QTreeWidgetItem(self.treeWidget)
        item.setText(0, file_name)
        item.setData(0, QtCore.Qt.UserRole, file_path)
        return item

    def setup_matplotlib_chinese(self):
        """配置matplotlib支持中文显示"""
        # 设置中文字体
        mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

    def change_range(self):
        comboBox_3 = self.comboBox_3.currentText()
        comboBox_4 = self.comboBox_4.currentText()
        if comboBox_3 == '最小值':
            self.lineEdit_3.setText(self.comboBox_3.currentText())
            self.lineEdit_3.setEnabled(False)
        else:
            self.lineEdit_3.setText('')
            self.lineEdit_3.setEnabled(True)
        if comboBox_4 == '最大值':
            self.lineEdit_4.setText(self.comboBox_4.currentText())
            self.lineEdit_4.setEnabled(False)
        else:
            self.lineEdit_4.setText('')
            self.lineEdit_4.setEnabled(True)

    def change_time_unit(self):
        text_5 = self.comboBox_5.currentText()
        text_6 = self.comboBox_6.currentText()
        text_7 = self.comboBox_7.currentText()
        self.time_unit_5 = UNIT_MAPPING[text_5]
        self.time_unit_6 = UNIT_MAPPING[text_6]
        self.time_unit_7 = UNIT_MAPPING[text_7]
    
    def fit_curve_model(self):
        text = self.comboBox.currentText()
        if text == '双指数':
            self.label_21.setText('I(t) = A1exp(-t/τ1) + A2exp(-t/τ2) + I0')
        if text == '单指数':
            self.label_21.setText('I(t) = Aexp(-t/τ) + I0')

    def comboBox_10_changed(self):
        text = self.comboBox_10.currentText()
        if text == '自动':
            self.lineEdit_8.setText('当整行都为数值时')
            self.lineEdit_8.setEnabled(False)
        if text == '行':
            self.lineEdit_8.setText('1')
            self.lineEdit_8.setEnabled(True)
    
    def restore_previous_settings(self):
        """恢复上一次的工作路径和文件列表"""
        # 恢复工作路径
        last_work_dir = self.settings.value("last_work_directory", "")
        if last_work_dir and os.path.exists(last_work_dir):
            self.pushButton.setText(last_work_dir)
        last_download_dir = self.settings.value("last_download_directory", "")
        if last_download_dir and os.path.exists(last_download_dir):
            self.pushButton_3.setText(last_download_dir)
        last_img_dir = self.settings.value("last_img_directory", "")

        
        # 恢复文件列表
        file_list = self.settings.value("last_file_list", [])
        if file_list:
            for file_path in file_list:
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    item = self._create_tree_item(file_path)
            
            # 更新按钮文本显示已选择文件数量
            selected_count = self.treeWidget.topLevelItemCount()
            self.pushButton_2.setText(f"已选择 {selected_count} 个文件")

    def set_download_directory(self):
        """打开文件夹选择对话框并更新lineEdit内容"""
        try:
            initial_dir = self.pushButton_3.text() or QDir.homePath()
            folder_path = QFileDialog.getExistingDirectory(
                self,
                "选择数据保存路径",
                initial_dir,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )

            
            if folder_path:
                self.pushButton_3.setText(folder_path)
                if self.validate_folder_path(folder_path):
                    print(f"有效路径: {folder_path}")
                    # 保存设置
                    self.save_current_settings()
                else:
                    print("警告：路径不可访问")
            
            self.download_directory = self.pushButton_3.text()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置保存目录时出错:\n{str(e)}")

    def set_img_directory(self):
        """打开文件夹选择对话框"""
        try:
            initial_dir = self.pushButton_4.text() or QDir.homePath()
            folder_path = QFileDialog.getExistingDirectory(
                self,
                "选择图片保存路径",
                initial_dir,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )

            if folder_path:
                self.pushButton_4.setText(folder_path)
                if self.validate_folder_path(folder_path):
                    print(f"有效路径: {folder_path}")
                    # 保存设置
                    self.save_current_settings()
                else:
                    print("警告：路径不可访问")
            self.img_directory = self.pushButton_4.text()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置图片目录时出错:\n{str(e)}")
    
    def save_current_settings(self):
        """保存当前的工作路径和文件列表到设置"""
        try:
            # 保存工作路径
            work_dir = self.pushButton.text()
            if work_dir and os.path.exists(work_dir):
                self.settings.setValue("last_work_directory", work_dir)
            download_dir = self.pushButton_3.text()
            if download_dir and os.path.exists(download_dir):
                self.settings.setValue("last_download_directory", download_dir)        

            
            # 保存文件列表
            file_list = []
            for i in range(self.treeWidget.topLevelItemCount()):
                item = self.treeWidget.topLevelItem(i)
                file_path = item.data(0, QtCore.Qt.UserRole)
                if file_path and os.path.exists(file_path):
                    file_list.append(file_path)
            
            self.settings.setValue("last_file_list", file_list)
        except Exception as e:
            print(f"保存设置时出错: {str(e)}")

    def set_work_directory(self):
        """打开文件夹选择对话框并更新lineEdit内容"""
        try:
            initial_dir = self.pushButton.text() or QDir.homePath()
            
            folder_path = QFileDialog.getExistingDirectory(
                self,
                "选择工作文件夹",
                initial_dir,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            
            if folder_path:
                self.pushButton.setText(folder_path)
                if self.validate_folder_path(folder_path):
                    print(f"有效路径: {folder_path}")
                    # 保存设置
                    self.save_current_settings()
                else:
                    print("警告：路径不可访问")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置工作目录时出错:\n{str(e)}")
                
    def save_one(self):
        """保存当前文件的衰减计算结果到txt文件"""
        
        try:
            # 检查是否有选中的文件
            current_item = self.treeWidget.currentItem()
            if current_item is None:
                QMessageBox.warning(self, "警告", "请先选择一个文件")
                return
            
            # 检查保存目录是否设置
            self.download_directory = self.pushButton_3.text()

            if not self.download_directory or not os.path.exists(self.download_directory):
                QMessageBox.warning(self, "警告", "请先设置有效的保存目录")
                return
            
            file_path = current_item.data(0, QtCore.Qt.UserRole)
            file_name = os.path.basename(file_path)
            
            # 创建进度对话框
            progress = self.create_progress_dialog("处理进度", "正在处理文件...", 1)
            self.update_progress(progress, 0, f"正在处理: {file_name}")
            
            print(f"文件路径: {file_path}")

            # 加载文件数据
            if not self.load_file_data(file_path):
                QMessageBox.warning(self, "错误", f"无法加载文件数据: {file_name}")
                progress.close()
                return
            
            # 调用calculate_decay，不显示内部进度对话框
            success = self.calculate_decay(current_item, show_progress=False)
            if not success:
                QMessageBox.warning(self, "错误", f"处理文件 {file_name} 失败")
                progress.close()
                return
            
            # 如果有标定数据，计算温度
            if (hasattr(self, 'temperature_calibration') and self.temperature_calibration and
                hasattr(self, 'lifetime_calibration') and self.lifetime_calibration):
                try:
                    avg_lifetime = self.calculate_average_lifetime()
                    if avg_lifetime is not None:
                        self.calculate_temperature_by_interpolation(avg_lifetime)
                except Exception as e:
                    print(f"保存前计算温度时出错: {str(e)}")
            
            # 收集拟合结果
            result_data = self._create_result_data(file_name)
            if not result_data:
                QMessageBox.warning(self, "警告", "当前文件拟合失败，无法保存结果")
                progress.close()
                return

            # 完成进度条
            self.update_progress(progress, 1, "正在保存结果...")
            
            # 设置输出文件名
            output_file = self._get_output_filename(file_name, self.download_directory)
            
            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                # 写入表头
                header = self._generate_header(result_data)
                f.write(header + '\n')
                
                # 写入数据
                line = self._format_result_line(result_data)
                f.write(line + '\n')
            
            progress.close()
            
            QMessageBox.information(self, "完成", f"结果已保存到: {output_file}")
            print(f"结果已保存到: {output_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存结果时出错: {str(e)}")
            traceback.print_exc()
        
    def save_all(self):
        """保存所有文件的衰减计算结果到txt文件 - 详细进度版本"""
        
        try:
            # 检查是否有文件需要处理
            file_count = self.treeWidget.topLevelItemCount()
            if file_count == 0:
                QMessageBox.warning(self, "警告", "没有可处理的文件")
                return
            
            # 检查保存目录是否设置
            self.download_directory = self.pushButton_3.text()

            if not self.download_directory or not os.path.exists(self.download_directory):
                QMessageBox.warning(self, "警告", "请先设置有效的保存目录")
                return
            
            # 创建自定义进度对话框
            progress_dialog = QtWidgets.QDialog(self)
            progress_dialog.setWindowTitle("处理进度")
            progress_dialog.setFixedSize(400, 150)
            progress_dialog.setModal(True)
            
            layout = QtWidgets.QVBoxLayout(progress_dialog)
            
            # 当前文件标签
            current_file_label = QtWidgets.QLabel("准备开始...")
            current_file_label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(current_file_label)
            
            # 进度条
            progress_bar = QtWidgets.QProgressBar()
            progress_bar.setRange(0, file_count)
            layout.addWidget(progress_bar)
            
            # 进度文本
            progress_text = QtWidgets.QLabel(f"0/{file_count}")
            progress_text.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(progress_text)
            
            # 取消按钮
            cancel_button = QtWidgets.QPushButton("取消")
            cancel_button.clicked.connect(progress_dialog.reject)
            layout.addWidget(cancel_button)
            
            progress_dialog.show()
            
            # 创建结果列表
            all_results = []
            cancelled = False
            
            # 遍历所有文件
            for i in range(file_count):
                # 检查是否取消
                if progress_dialog.wasCanceled():
                    cancelled = True
                    break
                
                item = self.treeWidget.topLevelItem(i)
                file_path = item.data(0, QtCore.Qt.UserRole)
                file_name = os.path.basename(file_path)
                self.file_path = file_path

                # 更新进度显示
                current_file_label.setText(f"正在处理: {file_name}")
                progress_bar.setValue(i)
                progress_text.setText(f"{i+1}/{file_count}")
                
                QtWidgets.QApplication.processEvents()

                print(f"文件路径: {self.file_path}")

                # 加载文件数据
                if not self.load_file_data(self.file_path):
                    print(f"跳过无法加载的文件: {file_name}")
                    continue
            
                # 调用calculate_decay并传递item参数，不显示内部进度对话框
                success = self.calculate_decay(item, show_progress=False)
                if not success:
                    print(f"处理文件 {file_name} 失败")
                    continue
                
                # 如果有标定数据，计算温度
                if (hasattr(self, 'temperature_calibration') and self.temperature_calibration and
                    hasattr(self, 'lifetime_calibration') and self.lifetime_calibration):
                    try:
                        avg_lifetime = self.calculate_average_lifetime()
                        if avg_lifetime is not None:
                            self.calculate_temperature_by_interpolation(avg_lifetime)
                    except Exception as e:
                        print(f"保存前计算温度时出错: {str(e)}")
                
                print(f"平均寿命: {avg_lifetime}")
                
                # 收集拟合结果
                result_data = self._create_result_data(file_name)
                if result_data:
                    all_results.append(result_data)
                    print(f"成功收集文件 {file_name} 的拟合结果")
                else:
                    print(f"收集文件 {file_name} 的拟合结果时出错")
                    continue
            
            progress_dialog.close()
            
            # 如果用户取消了操作
            if cancelled:
                QMessageBox.information(self, "提示", "操作已取消")
                return
            
            # 如果没有结果，则返回
            if not all_results:
                QMessageBox.warning(self, "警告", "没有可保存的结果")
                return
            
            # 设置输出文件名（添加时间后缀）
            timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            output_file = os.path.join(self.download_directory, f"decay_results_{timestamp}.txt")
            
            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                # 写入表头
                header = self._generate_header(all_results[0])
                f.write(header + '\n')
                
                # 写入数据
                for result in all_results:
                    line = self._format_result_line(result)
                    f.write(line + '\n')
            
            QMessageBox.information(self, "完成", f"结果已保存到: {output_file}")
            print(f"结果已保存到: {output_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存结果时出错: {str(e)}")
            traceback.print_exc()

    def _create_result_data(self, file_name):
        """创建结果数据结构"""
        if not (hasattr(self, 'result') and self.result and self.result.success):
            return None
        
        # 计算平均寿命
        avg_lifetime = None
        if hasattr(self, 'result') and self.result and self.result.success:
            try:
                avg_lifetime = self.calculate_average_lifetime()
            except Exception as e:
                print(f"计算平均寿命时出错: {str(e)}")
                avg_lifetime = None
        
        # 使用已计算的温度（如果存在）
        temperature = None
        if hasattr(self, 'temperature'):
            temperature = self.temperature
        
        result_data = {
            'file_name': file_name,
            'model_type': self.comboBox.currentText(),
            'params': self.result.params.copy() if self.result.params is not None else [],
            'param_names': self.result.param_names.copy() if self.result.param_names is not None else [],
            'sse': self.result.sse,
            'rsquare': self.result.rsquare,
            'adjrsquare': self.result.adjrsquare,
            'rmse': self.result.rmse,
            'success': self.result.success,
            'avg_lifetime': avg_lifetime,  # 新增平均寿命
            'temperature': temperature     # 温度
        }
        
        return result_data

    def _get_output_filename(self, base_name, directory):
        """生成输出文件名"""
        timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
        safe_name = os.path.splitext(base_name)[0]
        return os.path.join(directory, f"decay_result_{safe_name}_{timestamp}.txt")

    def _generate_header(self, sample_result):
        """生成表头"""
        header_parts = ["文件名", "模型类型"]
        
        # 添加系数估计的列名
        if sample_result['param_names']:
            for param_name in sample_result['param_names']:
                header_parts.append(f"系数_{param_name}")
        
        # 添加拟合优度统计的列名
        header_parts.extend(["SSE", "R平方", "调整R平方", "RMSE", "拟合成功"])
        
        # 添加平均寿命和温度列（最后两栏）
        header_parts.append("平均寿命(s)")
        
        # 只要有标定数据就添加温度列
        if (hasattr(self, 'temperature_calibration') and self.temperature_calibration and
            hasattr(self, 'lifetime_calibration') and self.lifetime_calibration):
            header_parts.append("温度(°C)")
        
        return "\t".join(header_parts)

    def _format_result_line(self, result):
        """格式化单行结果，使用科学计数法"""
        line_parts = [
            result['file_name'],
            result['model_type']
        ]
        
        # 添加系数估计的值（使用科学计数法）
        if result['params'] is not None:
            for param_value in result['params']:
                line_parts.append(f"{param_value:.5e}")
        
        # 添加拟合优度统计的值（使用科学计数法）
        line_parts.extend([
            f"{result['sse']:.5e}",
            f"{result['rsquare']:.5e}",
            f"{result['adjrsquare']:.5e}",
            f"{result['rmse']:.5e}",
            "是" if result['success'] else "否"  # 拟合成功状态
        ])
        
        # 添加平均寿命（使用科学计数法）
        if result.get('avg_lifetime') is not None:
            line_parts.append(f"{result['avg_lifetime']:.5e}")
        else:
            line_parts.append("N/A")
        
        # 添加温度值（如果有标定数据就包含这一列）
        if (hasattr(self, 'temperature_calibration') and self.temperature_calibration and
            hasattr(self, 'lifetime_calibration') and self.lifetime_calibration):
            if result.get('temperature') is not None:
                line_parts.append(f"{result['temperature']:.2f}")  # 温度保持2位小数
            else:
                line_parts.append("N/A")
        
        return "\t".join(line_parts)

    def select_target_files(self):
        """打开文件选择对话框，选择Excel/TXT/CSV文件并添加到treeWidget"""
        try:
            initial_dir = self.pushButton.text() if self.pushButton.text() else QDir.homePath()
            
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "选择目标文件",
                initial_dir,
                self._get_file_filters()
            )
            
            if file_paths:
                # 清空现有内容（可选，根据需要决定是否保留之前的内容）
                self.treeWidget.clear()
                
                for file_path in file_paths:
                    if self._validate_file_extension(file_path):
                        self._create_tree_item(file_path)
                    else:
                        QMessageBox.warning(self, "警告", f"忽略不支持的文件类型: {os.path.basename(file_path)}")
                
                # 更新按钮文本显示已选择文件数量
                selected_count = self.treeWidget.topLevelItemCount()
                self.pushButton_2.setText(f"已选择 {selected_count} 个文件")
                
                # 保存设置
                self.save_current_settings()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"选择文件时出错:\n{str(e)}")
        
    def _load_excel_data(self, file_path, column1, column2, data_start_row=None):
        """加载Excel数据"""
        try:
            if data_start_row is not None:
                data = pd.read_excel(file_path, skiprows=data_start_row).iloc[:, [column1, column2]]
            else:
                data = pd.read_excel(file_path).iloc[:, [column1, column2]]
            
            return self._clean_dataframe(data)
        except Exception as e:
            print(f"Excel读取失败: {e}")
            return None

    def _load_text_data_auto(self, file_path, column1, column2):
        """自动加载文本数据"""
        try:
            # 使用pandas读取，指定数据类型为数值
            df = pd.read_csv(file_path, header=None, dtype=str)  # 先读为字符串
            
            # 查找数据开始行
            data_start_row = 0
            for i in range(len(df)):
                try:
                    val1 = df.iloc[i, column1]
                    val2 = df.iloc[i, column2]
                    if pd.notna(val1) and pd.notna(val2) and val1.strip() and val2.strip():
                        # 尝试转换为浮点数
                        float(val1)
                        float(val2)
                        data_start_row = i
                        break
                except (ValueError, TypeError):
                    continue
            
            # 重新读取数据并转换为数值
            data = pd.read_csv(file_path, header=None, skiprows=data_start_row, 
                            usecols=[column1, column2], dtype=str)
            
            return self._clean_dataframe(data)
            
        except Exception as e:
            print(f"Pandas读取失败，尝试手动解析: {e}")
            return None

    def _clean_dataframe(self, df):
        """清理数据框"""
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        return df.values if not df.empty else None

    def load_file_data(self, file_path):
        """加载文件数据到缓存，确保数据为数值类型"""
        try:
            # 验证列设置
            try:
                column1 = int(self.lineEdit_9.text()) - 1
                column2 = int(self.lineEdit_10.text()) - 1
                if column1 < 0 or column2 < 0:
                    QMessageBox.warning(self, "警告", "列号必须为正整数")
                    return False
            except ValueError:
                QMessageBox.warning(self, "警告", "列号必须为有效数字")
                return False
            
            if self.comboBox_10.currentText() == '自动':
                if file_path.lower().endswith(('.txt', '.csv')):
                    data = self._load_text_data_auto(file_path, column1, column2)
                    if data is None:
                        data = self._load_file_manually(file_path, column1, column2, None)
                
                elif file_path.lower().endswith(('.xlsx', '.xls')):
                    data = self._load_excel_data(file_path, column1, column2)
            
            elif self.comboBox_10.currentText() == '行':
                try:
                    data_start_row = int(self.lineEdit_8.text()) - 1
                    if data_start_row < 0:
                        QMessageBox.warning(self, "警告", "起始行必须为正整数")
                        return False
                except ValueError:
                    QMessageBox.warning(self, "警告", "起始行必须为有效数字")
                    return False
                    
                if file_path.lower().endswith(('.txt', '.csv')):
                    data = self._load_file_manually(file_path, column1, column2, data_start_row)
                elif file_path.lower().endswith(('.xlsx', '.xls')):
                    data = self._load_excel_data(file_path, column1, column2, data_start_row)
            
            if data is not None:
                print(f"加载数据形状: {data.shape}, 数据类型: {data.dtype}")
                self.file_data_cache[file_path] = data
                return True
            else:
                QMessageBox.warning(self, "警告", f"无法从文件 {os.path.basename(file_path)} 中读取有效数据")
                return False
                
        except Exception as e:
            error_msg = f"加载文件 {file_path} 出错: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            traceback.print_exc()
            return False
         
    def _load_file_manually(self, file_path, column1, column2, data_start_row):
        """手动解析文件，确保数据类型正确"""
        try:
            valid_data = []
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            start_processing = False
            line_count = 0
            
            for line in lines:
                line_count += 1
                
                # 如果指定了起始行，检查是否到达起始行
                if data_start_row is not None:
                    if line_count <= data_start_row:
                        continue
                    start_processing = True
                else:
                    # 自动模式：检查是否是数据行
                    if not start_processing:
                        if self._is_data_line(line, column1, column2):
                            start_processing = True
                        else:
                            continue
                
                if start_processing:
                    # 处理数据行
                    processed_line = self._process_data_line(line, column1, column2)
                    if processed_line is not None:
                        valid_data.append(processed_line)
            
            if not valid_data:
                print(f"文件 {file_path} 中没有找到有效数据")
                return None
            
            # 转换为numpy数组并确保为浮点类型
            data = np.array(valid_data, dtype=float)
            print(f"手动加载数据形状: {data.shape}, 数据类型: {data.dtype}")
            
            return data
            
        except Exception as e:
            print(f"手动解析文件 {file_path} 出错: {str(e)}")
            return None

    def _is_data_line(self, line, column1, column2):
        """检查行是否是有效的数据行"""
        line = line.strip()
        if not line:
            return False
        
        parts = line.split(',')
        if len(parts) <= max(column1, column2):
            return False
        
        try:
            # 尝试转换两个列的值
            val1 = parts[column1].strip()
            val2 = parts[column2].strip()
            
            # 检查是否为空字符串
            if not val1 or not val2:
                return False
            
            float(val1)
            float(val2)
            return True
            
        except (ValueError, IndexError):
            return False

    def _process_data_line(self, line, column1, column2):
        """处理单行数据，返回[x, y]或None，确保数值类型"""
        line = line.strip()
        if not line:
            return None
        
        # 支持多种分隔符：逗号、制表符、空格
        separators = [',', '\t', ' ']
        parts = None
        
        for sep in separators:
            temp_parts = line.split(sep)
            if len(temp_parts) > max(column1, column2):
                parts = [p.strip() for p in temp_parts if p.strip()]  # 移除空字符串
                break
        
        if parts is None or len(parts) <= max(column1, column2):
            return None
        
        try:
            val1 = parts[column1]
            val2 = parts[column2]
            
            # 跳过空值
            if not val1 or not val2:
                return None
            
            # 移除可能的引号或其他非数字字符
            val1 = val1.replace('"', '').replace("'", "").strip()
            val2 = val2.replace('"', '').replace("'", "").strip()
            
            x_val = float(val1)
            y_val = float(val2)
            
            return [x_val, y_val]
            
        except (ValueError, IndexError) as e:
            # 打印无法转换的值以便调试
            print(f"无法转换的数据: '{parts[column1]}', '{parts[column2]}' - 错误: {e}")
            return None

    def create_progress_dialog(self, title, message, maximum=0):
        """创建进度对话框 - 修复动画显示"""
        progress = QtWidgets.QProgressDialog(message, "取消", 0, maximum, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)  # 立即显示
        
        if maximum == 0:  # 不确定进度
            progress.setCancelButton(None)  # 移除取消按钮
            
        # 强制显示
        progress.show()
        progress.setValue(0)
        QtWidgets.QApplication.processEvents()
        
        return progress

    def update_progress(self, progress, value, message=None):
        """更新进度 - 修复动画显示"""
        if message:
            progress.setLabelText(message)
        if value >= 0:
            progress.setValue(value)
        
        # 强制重绘以确保动画显示
        progress.repaint()
        QtWidgets.QApplication.processEvents()
        
        return not progress.wasCanceled()

    def data_range(self, x, y_data):
        """数据范围筛选"""
        try:
            # 确保输入是numpy数组
            x = np.asarray(x)
            y_data = np.asarray(y_data)
            
            # 初始化筛选条件（默认不筛选）
            x_mask = np.ones_like(x, dtype=bool)
            y_mask = np.ones_like(y_data, dtype=bool)

            # 处理时间范围筛选（x轴范围）
            if self.lineEdit_2.text():  # x_min
                x_min = float(self.lineEdit_2.text())
                x_mask &= (x >= x_min)

            if self.lineEdit_5.text():  # x_max
                x_max = float(self.lineEdit_5.text())
                x_mask &= (x <= x_max)
            
            # 应用时间筛选后的数据
            x_filtered = x[x_mask]
            y_filtered = y_data[x_mask]
            
            # 如果没有数据通过时间筛选，直接返回空数组
            if len(x_filtered) == 0:
                return np.array([]), np.array([])
            
            # 计算vrange（基于时间筛选后的数据）
            vrange = np.max(y_filtered) - np.min(y_filtered)
            
            # 处理最大值条件 - 仅保留从最大值点开始的数据（不移动x轴）
            if self.lineEdit_4.text():
                if self.lineEdit_4.text().lower() == '最大值':
                    max_idx = np.argmax(y_filtered)
                    x_filtered = x_filtered[max_idx:]
                    y_filtered = y_filtered[max_idx:]
                    # 已删除时间偏移调整代码
            
            # 处理最小值条件 - 保留到最小值点为止的数据
            if self.lineEdit_3.text():
                if self.lineEdit_3.text().lower() == '最小值':
                    min_idx = np.argmin(y_filtered)
                    x_filtered = x_filtered[:min_idx+1]
                    y_filtered = y_filtered[:min_idx+1]
            
            # 处理百分比范围条件
            if self.lineEdit_4.text() and self.lineEdit_4.text().lower() != '最大值':
                try:
                    y_max = float(self.lineEdit_4.text()) * vrange * 0.01
                    mask = (y_filtered <= y_max)
                    x_filtered = x_filtered[mask]
                    y_filtered = y_filtered[mask]
                except ValueError:
                    pass
            
            if self.lineEdit_3.text() and self.lineEdit_3.text().lower() != '最小值':
                try:
                    y_min = float(self.lineEdit_3.text()) * vrange * 0.01
                    mask = (y_filtered >= y_min)
                    x_filtered = x_filtered[mask]
                    y_filtered = y_filtered[mask]
                except ValueError:
                    pass
            
            # 保留您最初的调试信息
            print('def data_range')
            if len(y_filtered) > 0:
                print('y 起始值:', y_filtered[0])
            if len(x_filtered) > 0:
                print('x 起始值:', x_filtered[0])

            return x_filtered, y_filtered
        except Exception as e:
            print(f"数据范围筛选时出错: {str(e)}")
            return np.array([]), np.array([])

    def calculate_decay(self, item=None, show_progress=True):
        """计算衰减参数 - 支持有参数和无参数调用
        show_progress: 是否显示进度对话框，批量处理时不显示
        """
        progress = None
        if show_progress:
            # 创建进度对话框 - 仅当需要显示时创建
            progress = QtWidgets.QProgressDialog("正在计算衰减参数...", "取消", 0, 0, self)
            progress.setWindowTitle("处理中")
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(0)  # 立即显示
            progress.setCancelButton(None)  # 移除取消按钮，防止用户中断重要计算
            progress.show()
            
            # 强制显示进度对话框
            progress.setValue(0)
            
            # 处理事件，确保界面更新
            QtWidgets.QApplication.processEvents()
        
        try:
            # 如果没有传递item参数，使用当前选中的项目
            if item is None:
                current_item = self.treeWidget.currentItem()
                if current_item is None:
                    # 如果没有选中的项目，尝试使用第一个项目
                    if self.treeWidget.topLevelItemCount() > 0:
                        current_item = self.treeWidget.topLevelItem(0)
                    else:
                        QMessageBox.warning(self, "警告", "没有可处理的文件")
                        if progress:
                            progress.close()
                        return False
                file_path = current_item.data(0, QtCore.Qt.UserRole)
            else:
                # 确保item是QTreeWidgetItem类型，而不是布尔值
                if isinstance(item, bool):
                    # 如果是布尔值，使用当前选中的项目
                    current_item = self.treeWidget.currentItem()
                    if current_item is None:
                        if self.treeWidget.topLevelItemCount() > 0:
                            current_item = self.treeWidget.topLevelItem(0)
                        else:
                            QMessageBox.warning(self, "警告", "没有可处理的文件")
                            if progress:
                                progress.close()
                            return False
                    file_path = current_item.data(0, QtCore.Qt.UserRole)
                else:
                    file_path = item.data(0, QtCore.Qt.UserRole)
            
            file_name = os.path.basename(file_path)
            
            if progress:
                # 更新进度显示
                progress.setLabelText(f"正在处理: {file_name}\n请稍候...")
                progress.repaint()  # 强制重绘
                QtWidgets.QApplication.processEvents()
            
            print(f"处理文件: {file_path}")
            
            self.file_path = file_path

            if progress:
                progress.setLabelText("清理图形...")
                progress.repaint()
                QtWidgets.QApplication.processEvents()
            
            self.clear_figures()
            
            # 重置相关变量
            self.result = None
            self.lifetime = None

            if progress:
                progress.setLabelText("加载文件数据...")
                progress.repaint()
                QtWidgets.QApplication.processEvents()
            
            if file_path in self.file_data_cache:
                del self.file_data_cache[file_path]
                
            if not self.load_file_data(file_path):
                QMessageBox.warning(self, "错误", f"无法加载文件数据: {os.path.basename(file_path)}")
                if progress:
                    progress.close()
                return False
            
            data = self.file_data_cache[file_path]
            print(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
            
            # 数据验证
            if len(data.shape) != 2 or data.shape[1] < 2:
                QMessageBox.warning(self, "错误", f"数据格式不正确，期望至少2列，实际形状: {data.shape}")
                if progress:
                    progress.close()
                return False
            
            if len(data) == 0:
                QMessageBox.warning(self, "错误", "文件为空或没有有效数据")
                if progress:
                    progress.close()
                return False
            
            # 确保数据为数值类型
            if data.dtype.kind not in 'buifc':  # 检查是否为数值类型
                data = data.astype(float)
            
            t_data = data[:, 0]  # 第一列作为x轴数据
            y_data = data[:, 1]  # 第二列作为y轴数据
            
            # 安全的打印语句
            try:
                print(f"数据范围 - x: [{np.min(t_data):.3f}, {np.max(t_data):.3f}], y: [{np.min(y_data):.3f}, {np.max(y_data):.3f}]")
            except:
                print(f"数据范围 - x: [{np.min(t_data)}, {np.max(t_data)}], y: [{np.min(y_data)}, {np.max(y_data)}]")
            
            if progress:
                progress.setLabelText("绘制原始数据...")
                progress.repaint()
                QtWidgets.QApplication.processEvents()
            
            ax = self.figure.add_subplot(111)
            x_label = self.comboBox_7.currentText() if self.comboBox.currentText() else "X轴"
            
            ax.plot(t_data, y_data, 'b-', linewidth=1)
            ax.set_xlabel(f'时间/{x_label}')
            ax.set_ylabel("相对强度")
            ax.set_title(f'原始数据 - {os.path.basename(file_path)}')
            ax.grid(True)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            if progress:
                progress.setLabelText("进行曲线拟合...\n这可能需要一些时间")
                progress.repaint()
                QtWidgets.QApplication.processEvents()
            
            self._perform_fitting(data, file_path)
            
            if progress:
                progress.close()
            return True
            
        except Exception as e:
            if progress:
                progress.close()
            error_msg = f"处理文件时出错: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "错误", error_msg)
            return False

    def _perform_fitting(self, data, file_path):
        """执行拟合计算的内部方法"""
        try:
            Origional_x = data[:, 0]
            Origional_y_data = data[:, 1]
            
            # 创建新的拟合器实例，避免状态污染
            fitter = MATLABCurveFitter()

            # 配置拟合选项
            fitter = self._setup_fitter_options(fitter)

            # 获取模型类型并保存到实例变量中
            self.current_model_type = self.comboBox.currentText()
            models_to_test = self.current_model_type
            print(f"使用模型: {models_to_test}")

            # 数据范围筛选
            x, y_data = self.data_range(Origional_x, Origional_y_data)
            print(f"筛选后数据点数: {len(x)}")
            
            if len(x) < 2:
                QMessageBox.warning(self, "警告", "筛选后的数据点不足，无法进行拟合")
                return

            # 执行拟合
            result = fitter.fit_curve(x, y_data, models_to_test)
            self.result = result

            # 处理拟合结果
            if result.success:
                self._handle_fit_result(result, models_to_test, Origional_x, Origional_y_data, x, y_data, fitter)
            else:
                QMessageBox.warning(self, "警告", f"拟合失败: {result.message}")
                
        except Exception as e:
            error_msg = f"拟合计算时出错: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "错误", error_msg)

    def _setup_fitter_options(self, fitter):
        """设置拟合器选项"""
        try:
            robust_mapping = {
                'Off': RobustMethod.OFF,
                'LAR': RobustMethod.LAR,
                'Bisquare': RobustMethod.BISQUARE
            }

            algorithm_mapping = {
                'Levenberg-Marquardt': Algorithm.LEVENBERG_MARQUARDT,
                'Trust-Region': Algorithm.TRUST_REGION,
            }

            robust_method = robust_mapping.get(self.comboBox_8.currentText(), RobustMethod.OFF)
            algorithm_method = algorithm_mapping.get(self.comboBox_9.currentText(), Algorithm.LEVENBERG_MARQUARDT)
            
            fit_options = self.get_fit_options()
            
            fitter.set_options(
                Robust=robust_method,
                Algorithm=algorithm_method,
                **fit_options
            )
            
            return fitter
        except Exception as e:
            print(f"设置拟合器选项时出错: {str(e)}")
            return fitter

    def _handle_fit_result(self, result, models_to_test, Origional_x, Origional_y_data, x, y_data, fitter):
        """处理拟合结果"""
        try:
            # 确保模型类型信息被保存
            self.current_model_type = models_to_test
            
            if models_to_test == '单指数':
                b = result.params[1]
                lifetime = -1/b if b != 0 else float('inf')
                self.lifetime = lifetime
                self.lineEdit.setText(f"{lifetime:.8f}")
                print(f"单指数拟合结果: lifetime = {lifetime:.6f}")

            elif models_to_test == '双指数':
                # 修复参数排序逻辑，确保 t1 < t2
                a, t1, c, t2, e = result.params
                
                # 如果 t2 < t1，交换两个分量的参数
                if t2 < t1:
                    # 交换快速分量和慢速分量
                    a, c = c, a  # 交换振幅
                    t1, t2 = t2, t1  # 交换时间常数
                    print(f"参数已交换: t1={t1:.6f}, t2={t2:.6f}")
                
                # 更新界面显示（确保 t1 < t2）
                self.lineEdit_27.setText(f"{a:.5e}")  # 快速分量振幅
                self.lineEdit.setText(f"{t1:.5e}")    # 快速衰减时间常数 (t1)
                self.lineEdit_26.setText(f"{c:.5e}")  # 慢速分量振幅
                self.lineEdit_28.setText(f"{t2:.5e}") # 慢速衰减时间常数 (t2)
                self.lineEdit_29.setText(f"{e:.5e}")  # 基线
                
                # 同时更新result.params以确保保存时使用正确的顺序
                result.params = [a, t1, c, t2, e]
                
                print(f"双指数拟合结果: t1 = {t1:.5e}, t2 = {t2:.5e}, 振幅1 = {a:.5e}, 振幅2 = {c:.5e}")

            # 更新统计信息
            self.lineEdit_30.setText(f"{result.sse:.5e}")
            self.lineEdit_7.setText(f"{result.rsquare:.5e}")
            self.lineEdit_31.setText(f"{result.adjrsquare:.5e}")
            self.lineEdit_32.setText(f"{result.rmse:.5e}")
            self.lineEdit_33.setText(f"{'是' if result.success else '否'}")

            # 绘制结果 - 保持原有绘图逻辑
            model_def = fitter._get_model_definition(models_to_test)
            if model_def:
                # 使用原始的参数顺序进行绘图，不应用交换
                y_pred = model_def['function'](x, *result.params)
                self.fig1fig2_plot(Origional_x, Origional_y_data, x, y_data, y_pred, models_to_test)
            
            print("拟合完成:", result)
            
        except Exception as e:
            error_msg = f"处理拟合结果时出错: {str(e)}"
            print(error_msg)
            traceback.print_exc()

    def fig1fig2_plot(self, Origional_x, Origional_y_data, x, y_data, y_pred, models_to_test):
        """绘制拟合结果图形"""
        try:
            # 绘制图形
            self.figure2.clear()
            ax2 = self.figure2.add_subplot(111)

            # 绘制所有数据点
            ax2.scatter(Origional_x, Origional_y_data, alpha=0.3, color='gray', s=5)
            # 高亮显示用于拟合的数据点
            ax2.scatter(x, y_data, alpha=0.8, color='blue', s=10)
            
            ax2.plot(x, y_pred, 'r-', label=f'{models_to_test}拟合', linewidth=2)
            ax2.legend()
            ax2.set_title(f'{models_to_test}模型拟合结果')
            ax2.set_xlabel('时间')
            ax2.set_ylabel("相对强度")
            ax2.grid(True, alpha=0.3)
            
            # 自动调整布局
            self.figure2.tight_layout()
            self.canvas2.draw()

            # 绘制残差
            self.figure3.clear()
            ax3 = self.figure3.add_subplot(111)
            
            ax3.plot(x, self.result.residuals, 'ro-', alpha=0.6, markersize=2)
            ax3.axhline(y=0, color='k', linestyle='--')
            ax3.set_title('残差图')
            ax3.set_xlabel('x')
            ax3.set_ylabel('残差')
            ax3.grid(True, alpha=0.3)
            
            # 自动调整布局
            self.figure3.tight_layout()
            self.canvas3.draw()
            
            self.calculate_lifetime()
        except Exception as e:
            print(f"绘图时出错: {str(e)}")
            
    def calculate_lifetime(self):
        """使用线性插值和两侧外推方法计算温度，基于平均寿命"""
        try:
            # 检查是否有拟合结果
            if self.result is None or not self.result.success:
                QMessageBox.warning(self, "警告", "请先进行成功的拟合计算")
                return
            
            # 检查标定数据（修改为检查是否为空列表）
            if not hasattr(self, 'temperature_calibration') or not self.temperature_calibration:
                QMessageBox.warning(self, "警告", "请先导入标定数据")
                return
            
            if not hasattr(self, 'lifetime_calibration') or not self.lifetime_calibration:
                QMessageBox.warning(self, "警告", "请先导入标定数据")
                return
            
            # 计算平均寿命
            avg_lifetime = self.calculate_average_lifetime()
            if avg_lifetime is None:
                QMessageBox.warning(self, "警告", "无法计算平均寿命")
                return
            
            # 使用线性插值和两侧外推计算温度（这会设置self.temperature）
            temperature = self.calculate_temperature_by_interpolation(avg_lifetime)
            if temperature is None:
                QMessageBox.warning(self, "警告", "温度计算失败")
                return
            
            # 显示结果
            self.display_temperature_result(temperature, avg_lifetime)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算寿命温度时出错: {str(e)}")
            traceback.print_exc()

    def calculate_average_lifetime(self):
        """计算平均寿命 (a*t1^2 + c*t2^2) / (a*t1 + c*t2)"""
        try:
            model_type = self.comboBox.currentText()
            
            if model_type == '单指数':
                # 单指数模型：寿命就是衰减常数
                tau = self.result.params[1]  # b参数
                print(f"单指数寿命: {tau:.5e} s")
                return tau
                
            elif model_type == '双指数':
                # 双指数模型：计算加权平均寿命
                # 参数顺序: [a, t1, c, t2, e]
                a = self.result.params[0]  # 快速分量振幅
                t1 = self.result.params[1]  # 快速衰减时间常数
                c = self.result.params[2]  # 慢速分量振幅
                t2 = self.result.params[3]  # 慢速衰减时间常数
                
                # 计算平均寿命: (a*t1^2 + c*t2^2) / (a*t1 + c*t2)
                numerator = a * (t1 ** 2) + c * (t2 ** 2)
                denominator = a * t1 + c * t2
                
                if denominator == 0:
                    print("警告：分母为零，无法计算平均寿命")
                    return None
                    
                avg_lifetime = numerator / denominator
                print(f"双指数平均寿命计算:")
                print(f"  a = {a:.5e}, t1 = {t1:.5e} s")
                print(f"  c = {c:.5e}, t2 = {t2:.5e} s")
                print(f"  分子: {numerator:.5e}, 分母: {denominator:.5e}")
                print(f"  平均寿命: {avg_lifetime:.5e} s")
                return avg_lifetime
                
            else:
                print(f"不支持的模型类型: {model_type}")
                return None
                
        except Exception as e:
            print(f"计算平均寿命时出错: {str(e)}")
            traceback.print_exc()
            return None

    def calculate_temperature_by_interpolation(self, lifetime):
        """使用线性插值和两侧外推计算温度，并保存到self.temperature"""
        try:
            # 确保标定数据是numpy数组（单位：秒s）
            lifetimes = np.array(self.lifetime_calibration)
            temperatures = np.array(self.temperature_calibration)
            
            # 对数据进行排序
            sorted_indices = np.argsort(lifetimes)
            sorted_lifetimes = lifetimes[sorted_indices]
            sorted_temperatures = temperatures[sorted_indices]
            
            # 使用科学计数法显示（单位：秒）
            print(f"标定数据 - 寿命范围: [{min(sorted_lifetimes):.5e}, {max(sorted_lifetimes):.5e}] s")
            print(f"标定数据 - 温度范围: [{min(sorted_temperatures):.2f}, {max(sorted_temperatures):.2f}] °C")
            print(f"当前寿命: {lifetime:.5e} s")
            
            # 检查寿命值是否在标定数据范围内
            if lifetime <= sorted_lifetimes[0]:
                # 左侧外推：使用前两个点
                x0, x1 = sorted_lifetimes[0], sorted_lifetimes[1]
                y0, y1 = sorted_temperatures[0], sorted_temperatures[1]
                slope = (y1 - y0) / (x1 - x0)
                temperature = y0 + slope * (lifetime - x0)
                method = "左侧外推"
                
            elif lifetime >= sorted_lifetimes[-1]:
                # 右侧外推：使用最后两个点
                x0, x1 = sorted_lifetimes[-2], sorted_lifetimes[-1]
                y0, y1 = sorted_temperatures[-2], sorted_temperatures[-1]
                slope = (y1 - y0) / (x1 - x0)
                temperature = y1 + slope * (lifetime - x1)
                method = "右侧外推"
                
            else:
                # 线性插值：在范围内
                temperature = np.interp(lifetime, sorted_lifetimes, sorted_temperatures)
                method = "线性插值"
            
            # 保存温度到self.temperature
            self.temperature = temperature
            
            print(f"计算方法: {method}, 计算温度: {temperature:.2f} °C")
            return temperature
            
        except Exception as e:
            print(f"温度插值计算时出错: {str(e)}")
            traceback.print_exc()
            return None

    def display_temperature_result(self, temperature, avg_lifetime):
        """显示温度计算结果，保持现有textEdit格式"""
        try:
            model_type = self.comboBox.currentText()
            
            # 构建显示文本
            result_text = f"""
            <div align="center">
                <font color="red" size="9">{temperature:.2f} ℃</font>
            </div>
            """
            
            self.textEdit.setHtml(result_text)
            
            # 同时在控制台输出详细信息
            print(f"温度计算结果:")
            print(f"  模型类型: {model_type}")
            print(f"  平均寿命: {avg_lifetime:.6f}")
            print(f"  计算温度: {temperature:.2f} °C")
            print(f"  拟合优度 R²: {self.result.rsquare:.4f}")
            
        except Exception as e:
            print(f"显示温度结果时出错: {str(e)}")
            # 如果出错，至少显示温度值
            self.textEdit.setHtml(f'<div align="center"><font color="red" size="9">{temperature:.2f} ℃</font></div>')
    
    def validate_folder_path(self, path):
        """验证文件夹路径是否有效"""
        return QDir(path).exists()

    def clear_figures(self):
        """清除所有图形"""
        for figure in [self.figure, self.figure2, self.figure3]:
            figure.clear()

    def cleanup(self):
        """清理资源"""
        self.file_data_cache.clear()
        self.clear_figures()
        self.save_current_settings()

    def closeEvent(self, event):
        """重写关闭事件"""
        try:
            self.cleanup()
            event.accept()
        except Exception as e:
            print(f"关闭程序时出错: {str(e)}")
            event.accept()

if __name__ == '__main__':
    try:
        QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        app = QApplication(sys.argv)
        myWin = mainWindow()
        myWin.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序启动时出错: {str(e)}")
        traceback.print_exc()