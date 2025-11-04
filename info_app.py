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
        """
        使用预定义模型进行拟合
        """
        models = {
            'linear': {
                'function': lambda x, a, b: a * x + b,
                'param_names': ['a', 'b'],
                'description': 'y = a*x + b'
            },
            '单指数': {
                'function': lambda x, a, b, c: a * np.exp(b * x) + c,
                'param_names': ['a', 'b', 'c'],
                'description': 'y = a*exp(b*x) + c'
            },
            '双指数': {
                'function': lambda x, a, b, c, d, e: a * np.exp(b * x) + c * np.exp(d * x) + e,
                'param_names': ['a', 'b', 'c', 'd', 'e'],
                'description': 'y = a*exp(-x/b) + c*exp(-x/d) + e'
            },
            'polynomial2': {
                'function': lambda x, a, b, c: a * x**2 + b * x + c,
                'param_names': ['a', 'b', 'c'],
                'description': 'y = a*x² + b*x + c'
            },
        }
        
        return models.get(model_type)

    def _get_default_start_point(self, model_type, x, y):
        """获取更合理的默认起始点"""
        if model_type == 'linear':
            # 线性回归计算斜率和截距
            if len(x) > 1:
                slope = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 1.0
                intercept = y[0] - slope * x[0]
                return [slope, intercept]
            return [1.0, 0.0]
        
        elif model_type == '单指数':
            # 单指数起始点
            y_range = np.max(y) - np.min(y)
            if y_range == 0:
                y_range = 1.0
            return [y_range, -0.1, np.min(y)]
        
        elif model_type == '双指数':
            # 双指数起始点 - 更保守的估计
            y_range = np.max(y) - np.min(y)
            if y_range == 0:
                y_range = 1.0
            return [
                y_range * 0.7,  # a
                -0.1,           # b (衰减速率1)
                y_range * 0.3,  # c  
                -0.01,          # d (衰减速率2)
                np.min(y)       # e (基线)
            ]
        
        elif model_type == 'polynomial2':
            return [0.0, 0.0, np.mean(y)]
        
        else:
            model_def = self._get_model_definition(model_type)
            if model_def:
                return [1.0] * len(model_def['param_names'])
            return [1.0]
        
    def _fit_implementation(self, x, y, func, initial_guess, param_names):
        """拟合实现核心"""
        result = MATLABFitResult()
        result.param_names = param_names
        
        # 稳健拟合迭代
        weights = self.options.Weights
        robust_weights = np.ones_like(y)
        prev_params = np.array(initial_guess)
        
        for robust_iter in range(50):  # MATLAB最大稳健迭代
            # 配置最小二乘选项
            lsq_options = {
                'method': 'trf' if self.options.Algorithm == Algorithm.TRUST_REGION else 'lm',
                'max_nfev': self.options.MaxFunEvals,
                'ftol': self.options.TolFun,
                'xtol': self.options.TolX,
                'gtol': self.options.TolX,  # MATLAB中gtol通常等于xtol
                'x_scale': 'jac',
                'verbose': 0
            }
            
            # 设置边界
            if self.options.Algorithm == Algorithm.TRUST_REGION and \
               (np.isfinite(self.options.Lower).any() or np.isfinite(self.options.Upper).any()):
                bounds = (np.full_like(initial_guess, self.options.Lower), 
                         np.full_like(initial_guess, self.options.Upper))
                lsq_options['bounds'] = bounds
            
            try:
                # 目标函数
                def objective(params):
                    y_pred = func(x, *params)
                    residuals = y - y_pred
                    
                    # 应用稳健权重
                    if self.options.Robust != RobustMethod.OFF and robust_iter > 0:
                        residuals = residuals * np.sqrt(robust_weights)
                    
                    # 应用用户权重
                    if weights is not None:
                        residuals = residuals * np.sqrt(weights)
                    
                    return residuals
                
                # 执行拟合
                lsq_result = least_squares(objective, initial_guess, **lsq_options)
                
                result.params = lsq_result.x
                result.success = lsq_result.success
                result.message = lsq_result.message
                result.iterations = lsq_result.nfev
                result.funcCount = lsq_result.nfev
                result.jacobian = lsq_result.jac
                
                # 计算预测值和残差
                y_pred = func(x, *result.params)
                residuals = y - y_pred
                
                # 更新稳健权重
                if self.options.Robust != RobustMethod.OFF:
                    new_robust_weights = self._robust_weight_function(residuals, self.options.Robust)
                    
                    # 检查稳健收敛
                    if robust_iter > 0 and np.max(np.abs(new_robust_weights - robust_weights)) < 1e-6:
                        break
                    
                    robust_weights = new_robust_weights
                else:
                    break
                    
                # 检查参数收敛
                if np.max(np.abs(result.params - prev_params)) < self.options.TolX:
                    break
                    
                prev_params = result.params.copy()
                
            except Exception as e:
                result.success = False
                result.message = f"拟合失败: {str(e)}"
                break
        
        # 计算拟合优度
        y_pred = func(x, *result.params)
        residuals = y - y_pred
        
        # 应用最终权重计算统计量
        final_weights = weights
        if self.options.Robust != RobustMethod.OFF:
            if final_weights is None:
                final_weights = robust_weights
            else:
                final_weights = final_weights * robust_weights
        
        sse, rsquare, dfe, adjrsquare, rmse = self._calculate_goodness_of_fit(
            y, y_pred, residuals, len(param_names), final_weights)
        
        result.sse = sse
        result.rsquare = rsquare
        result.dfe = dfe
        result.adjrsquare = adjrsquare
        result.rmse = rmse
        result.residuals = residuals
        
        return result
    
class mainWindow(QMainWindow, main_window):
    def __init__(self, parent=None, product_names=None):
        super(mainWindow, self).__init__(parent)
        self.setupUi(self)
        
        self.temperature_calibration=[50,100,150,200,250,300,350,400,450,500,550,600]
        self.lifetime_calibration=[3.646040506,3.212977315,2.842532971,2.519596718,2.236712857,1.981563617,1.710492252,1.280798396,0.688913014,0.379551499,0.321295208,0.317258355]

        # 初始化QSettings
        self.settings = QSettings("YourCompany", "PMT_Analysis")
        
        self.setup_matplotlib_chinese()

        self.treeWidget.itemClicked.connect(self.click_file)
        self.treeWidget.itemClicked.connect(self.calculate_decay)
        self.lineEdit_8.setEnabled(False)
        self.comboBox_10.currentIndexChanged.connect(self.comboBox_10_changed)
        self.comboBox.currentIndexChanged.connect(self.fit_curve_model)
        self.file_path = None
        self.img_directory = None
        self.download_directory = None

        #设置文件夹
        self.pushButton.clicked.connect(self.set_work_directory)
        self.pushButton_2.clicked.connect(self.select_target_files)
        self.pushButton_3.clicked.connect(self.set_download_directory)
        self.pushButton_4.clicked.connect(self.set_img_directory)
        
        # 下载按钮
        self.pushButton_5.clicked.connect(self.save_all)
        # self.pushButton_7.clicked.connect(self.save_one)
        

        # 初始化treeWidget设置
        self.treeWidget.setHeaderLabels(["文件列表"])
        self.treeWidget.setColumnCount(1)

        # 初始化matplotlib图形
        self.figure = Figure()
        self.canvas = FC(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # 将matplotlib组件添加到graphicsView
        layout = QtWidgets.QVBoxLayout(self.graphicsView)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # 初始化第二个matplotlib图形 - 第二个图形视图
        self.figure2 = Figure()
        self.canvas2 = FC(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        
        # 将matplotlib组件添加到graphicsView2
        layout2 = QtWidgets.QVBoxLayout(self.graphicsView_2)
        layout2.addWidget(self.toolbar2)
        layout2.addWidget(self.canvas2)
        
        # 初始化第三个matplotlib图形 - 第三个图形视图
        self.figure3 = Figure()
        self.canvas3 = FC(self.figure3)
        self.toolbar3 = NavigationToolbar(self.canvas3, self)
        
        # 将matplotlib组件添加到graphicsView3
        layout3 = QtWidgets.QVBoxLayout(self.graphicsView_3)
        layout3.addWidget(self.toolbar3)
        layout3.addWidget(self.canvas3)
        
        # 存储当前选中的文件数据
        self.file_data_cache = {}
        
        self.lifetime = None
        
        # 恢复上一次的设置
        self.restore_previous_settings()
        
        # 单位设置
        self.comboBox_5.currentTextChanged.connect(self.change_time_unit)
        self.comboBox_6.currentTextChanged.connect(self.change_time_unit)
        self.comboBox_7.currentTextChanged.connect(self.change_time_unit)
        self.comboBox_11.currentTextChanged.connect(self.change_time_unit)


        # 寿命、温度计算
        self.comboBox.currentTextChanged.connect(self.calculate_decay)   
        self.comboBox_2.currentTextChanged.connect(self.calculate_decay)   
        self.comboBox_3.currentTextChanged.connect(self.calculate_decay)
        self.comboBox_4.currentTextChanged.connect(self.calculate_decay)        
        self.comboBox_5.currentTextChanged.connect(self.calculate_decay)
        self.comboBox_6.currentTextChanged.connect(self.calculate_decay)
        self.comboBox_7.currentTextChanged.connect(self.calculate_decay)
        self.comboBox_8.currentTextChanged.connect(self.calculate_decay)
        self.comboBox_9.currentTextChanged.connect(self.calculate_decay)
        self.comboBox_10.currentTextChanged.connect(self.calculate_decay)
        self.lineEdit_2.editingFinished.connect(self.calculate_decay)
        self.lineEdit_3.editingFinished.connect(self.calculate_decay)
        self.lineEdit_4.editingFinished.connect(self.calculate_decay)
        self.lineEdit_5.editingFinished.connect(self.calculate_decay)
        self.lineEdit_6.editingFinished.connect(self.calculate_decay)
        self.lineEdit_8.editingFinished.connect(self.calculate_decay)
        self.lineEdit_9.editingFinished.connect(self.calculate_decay)
        self.lineEdit_10.editingFinished.connect(self.calculate_decay)
        self.lineEdit_11.editingFinished.connect(self.calculate_decay)
        self.lineEdit_12.editingFinished.connect(self.calculate_decay)
        self.lineEdit_13.editingFinished.connect(self.calculate_decay)
        self.lineEdit_14.editingFinished.connect(self.calculate_decay)
        self.lineEdit_15.editingFinished.connect(self.calculate_decay)
        self.lineEdit_16.editingFinished.connect(self.calculate_decay)
        self.lineEdit_17.editingFinished.connect(self.calculate_decay)
        self.lineEdit_18.editingFinished.connect(self.calculate_decay)
        self.lineEdit_19.editingFinished.connect(self.calculate_decay)
        self.lineEdit_20.editingFinished.connect(self.calculate_decay)
        self.lineEdit_21.editingFinished.connect(self.calculate_decay)
        self.lineEdit_22.editingFinished.connect(self.calculate_decay)
        self.lineEdit_23.editingFinished.connect(self.calculate_decay)
        self.lineEdit_24.editingFinished.connect(self.calculate_decay)
        self.lineEdit_25.editingFinished.connect(self.calculate_decay)
        
    def change_time_unit(self):
        unit_mapping = {'s':1, 'ms':1000, 'μs':1000000, 'ns':1000000000}
        
        text_5 = self.comboBox_5.currentText()
        text_6 = self.comboBox_6.currentText()
        text_7 = self.comboBox_7.currentText()
        text_11 = self.comboBox_11.currentText()
        self.time_unit_5 = unit_mapping[text_5]
        self.time_unit_6 = unit_mapping[text_6]
        self.time_unit_7 = unit_mapping[text_7]
        self.time_unit_11 = unit_mapping[text_11]
    
    def fit_curve_model(self):
        text = self.comboBox.currentText()
        if text == '双指数':
            self.label_21.setText('I(t) = Aexp(-t/τ1) + Bexp(-t/τ2) + C')
        if text == '单指数':
            self.label_21.setText('I(t) = I0 exp(-t/τ) + C')


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
        if last_img_dir and os.path.exists(last_img_dir):
            self.pushButton_4.setText(last_img_dir)
        
        # 恢复文件列表
        file_list = self.settings.value("last_file_list", [])
        if file_list:
            for file_path in file_list:
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    item = QTreeWidgetItem(self.treeWidget)
                    item.setText(0, file_name)
                    item.setData(0, QtCore.Qt.UserRole, file_path)
            
            # 更新按钮文本显示已选择文件数量
            selected_count = self.treeWidget.topLevelItemCount()
            self.pushButton_2.setText(f"已选择 {selected_count} 个文件")

    def set_download_directory(self):
        """打开文件夹选择对话框并更新lineEdit内容"""
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

    def set_img_directory(self):
        """打开文件夹选择对话框"""
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
    
    def save_current_settings(self):
        """保存当前的工作路径和文件列表到设置"""
        # 保存工作路径
        work_dir = self.pushButton.text()
        if work_dir and os.path.exists(work_dir):
            self.settings.setValue("last_work_directory", work_dir)
        download_dir = self.pushButton_3.text()
        if download_dir and os.path.exists(download_dir):
            self.settings.setValue("last_download_directory", download_dir)        
        img_dir = self.pushButton_4.text()
        if img_dir and os.path.exists(img_dir):
            self.settings.setValue("last_img_directory", img_dir)        
        
        # 保存文件列表
        file_list = []
        for i in range(self.treeWidget.topLevelItemCount()):
            item = self.treeWidget.topLevelItem(i)
            file_path = item.data(0, QtCore.Qt.UserRole)
            if file_path and os.path.exists(file_path):
                file_list.append(file_path)
        
        self.settings.setValue("last_file_list", file_list)

    def setup_matplotlib_chinese(self):
        """配置matplotlib支持中文显示"""
        # 设置中文字体
        mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

    def set_work_directory(self):
        """打开文件夹选择对话框并更新lineEdit内容"""
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

    def save_all(self):
        """保存所有文件的衰减计算结果到txt文件"""
        
        try:
            # 检查是否有文件需要处理
            if self.treeWidget.topLevelItemCount() == 0:
                QMessageBox.warning(self, "警告", "没有可处理的文件")
                return
            
            # 检查保存目录是否设置
            self.download_directory = self.pushButton_3.text()

            if not self.download_directory or not os.path.exists(self.download_directory):
                QMessageBox.warning(self, "警告", "请先设置有效的保存目录")
                return
            
            # 创建结果列表
            results = []
            
            # 遍历所有文件
            for i in range(self.treeWidget.topLevelItemCount()):
                print(i)
                
                if not hasattr(self, 'loading_label'):
                    self.loading_label = QtWidgets.QLabel(self)
                    self.loading_label.setAlignment(QtCore.Qt.AlignCenter)
                    self.loading_label.setStyleSheet("""
                        QLabel {
                            background-color: rgba(0, 0, 0, 150);
                            color: white;
                            font-size: 16px;
                            border-radius: 10px;
                            padding: 20px;
                        }
                    """)
                    self.loading_label.setFixedSize(200, 100)
                    self.loading_label.move(
                        self.width()//2 - 100, 
                        self.height()//2 - 50
                    )
                
                self.loading_label.show()
                self.loading_label.raise_()
                
                # 设置动画文本
                self.loading_text = f"处理{i}中..."
                
                item = self.treeWidget.topLevelItem(i)
                file_path = item.data(0, QtCore.Qt.UserRole)
                file_name = os.path.basename(file_path)
                self.file_path = file_path

                print(self.file_path)

                # 加载文件数据
                if not self.load_file_data(self.file_path):
                    print(f"跳过无法加载的文件: {file_name}")
                    continue
            
                self.load_file_data(self.file_path)
                self.calculate_decay()
                
                # 计算衰减参数
                try:
                    results.append(self.lifetime)

                except Exception as e:
                    print(f"处理文件 {self.file_path} 时出错: {str(e)}")
                    traceback.print_exc()
                    continue
            
            if hasattr(self, 'loading_label'):
                self.loading_label.hide()
            
            # 如果没有结果，则返回
            if not results:
                QMessageBox.warning(self, "警告", "没有可保存的结果")
                return
            
            # 设置输出文件名
            timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            output_file = os.path.join(self.download_directory, f"decay_results_{timestamp}.txt")
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write('results\n')

            QMessageBox.information(self, "完成", f"结果已保存到: {output_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存结果时出错: {str(e)}")
            traceback.print_exc()
    
    def select_target_files(self):
        """打开文件选择对话框，选择Excel/TXT/CSV文件并添加到treeWidget"""
        initial_dir = self.pushButton.text() if self.pushButton.text() else QDir.homePath()
        
        file_filter = "Supported Files (*.xlsx *.xls *.txt *.csv);;" \
                     "Excel Files (*.xlsx *.xls);;" \
                     "Text Files (*.txt);;" \
                     "CSV Files (*.csv);;" \
                     "All Files (*.*)"
        
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择目标文件",
            initial_dir,
            file_filter
        )
        
        if file_paths:
            # 清空现有内容（可选，根据需要决定是否保留之前的内容）
            self.treeWidget.clear()
            
            for file_path in file_paths:
                if file_path.lower().endswith(('.xlsx', '.xls', '.txt', '.csv')):
                    # 创建树形项目并添加到treeWidget
                    file_name = os.path.basename(file_path)
                    item = QTreeWidgetItem(self.treeWidget)
                    item.setText(0, file_name)
                    
                    # 可选：将完整路径存储在item中
                    item.setData(0, QtCore.Qt.UserRole, file_path)
                else:
                    QMessageBox.warning(self, "警告", f"忽略不支持的文件类型: {os.path.basename(file_path)}")
            
            # 更新按钮文本显示已选择文件数量
            selected_count = self.treeWidget.topLevelItemCount()
            self.pushButton_2.setText(f"已选择 {selected_count} 个文件")
            
            # 保存设置
            self.save_current_settings()
        
        
    def load_file_data(self, file_path):
        """加载文件数据到缓存，处理空字符串和格式问题"""
        try:
            column1 = int(self.lineEdit_9.text()) - 1
            column2 = int(self.lineEdit_10.text()) - 1
            
            if self.comboBox_10.currentText() == '自动':
                if file_path.lower().endswith(('.txt', '.csv')):
                    # 方法1: 使用pandas读取，自动处理空值
                    try:
                        df = pd.read_csv(file_path, header=None)
                        
                        # 查找数据开始行
                        data_start_row = 0
                        for i in range(len(df)):
                            try:
                                # 检查该行是否包含数值数据
                                val1 = df.iloc[i, column1]
                                val2 = df.iloc[i, column2]
                                if pd.notna(val1) and pd.notna(val2):
                                    float(val1)
                                    float(val2)
                                    data_start_row = i
                                    break
                            except (ValueError, TypeError):
                                continue
                        
                        # 读取数据
                        data = pd.read_csv(file_path, header=None, skiprows=data_start_row, 
                                        usecols=[column1, column2])
                        
                        # 清理数据：移除包含NaN或空值的行
                        data = data.dropna()
                        
                        # 转换为numpy数组
                        data = data.values
                        self.file_data_cache[file_path] = data
                        return True
                        
                    except Exception as e:
                        print(f"Pandas读取失败，尝试手动解析: {e}")
                        # 方法2: 手动解析文件
                        return self._load_file_manually(file_path, column1, column2, None)
                
                elif file_path.lower().endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(file_path).values
                    self.file_data_cache[file_path] = data
                    return True
            
            elif self.comboBox_10.currentText() == '行':
                data_start_row = int(self.lineEdit_8.text()) - 1
                if file_path.lower().endswith(('.txt', '.csv')):
                    return self._load_file_manually(file_path, column1, column2, data_start_row)
                elif file_path.lower().endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(file_path, skiprows=data_start_row).values
                    self.file_data_cache[file_path] = data
                    return True
            
            return False
            
        except Exception as e:
            print(f"加载文件 {file_path} 出错: {str(e)}")
            traceback.print_exc()
            return False

    def click_file(self, item):
        """点击treeWidget中的文件时触发"""
        file_path = item.data(0, QtCore.Qt.UserRole)
        self.file_path = file_path

        if file_path not in self.file_data_cache:
            if not self.load_file_data(file_path):
                QMessageBox.warning(self, "错误", f"无法加载文件数据: {os.path.basename(file_path)}")
                return
        
        data = self.file_data_cache[file_path]
        
        try:
            if len(data.shape) == 1 or data.shape[1] < 2:
                raise ValueError("数据列数不足，无法绘制图形")
            
            t_data = data[:, 0]  # 第一列作为x轴数据
            y_data = data[:, 1]  # 第二列作为y轴数据
            
            if len(t_data) != len(y_data):
                raise ValueError("x轴和y轴数据长度不一致")
            
            # 绘制图形
            self.figure.clear()
            self.figure2.clear()
            self.figure3.clear()
            self.canvas2.draw()
            self.canvas3.draw()

            ax = self.figure.add_subplot(111)
            
            # 获取comboBox当前文本作为x轴标签
            x_label = self.comboBox_7.currentText() if self.comboBox.currentText() else "X轴"
            
            ax.plot(t_data, y_data, 'b-')
            ax.set_xlabel(f'时间/{x_label}')
            ax.set_ylabel("相对强度")
            ax.set_title(os.path.basename(file_path))
            
            # 自动调整坐标轴范围
            ax.relim()
            ax.autoscale_view()
            
            # 添加网格线
            ax.grid(True)
            
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"绘图时出错: {str(e)}")
            traceback.print_exc()

    def closeEvent(self, event):
        """重写关闭事件，保存当前设置"""
        self.save_current_settings()
        event.accept()

    def data_range(self, x, y_data):
        # 计算 vrange（假设 y_data 是 numpy 数组）
        vrange = np.average(y_data[:1000]) + np.average(y_data[-1000:])

        # 初始化筛选条件（默认不筛选）
        x_mask = np.ones_like(x, dtype=bool)  # 默认全部为True（不筛选）
        y_mask = np.ones_like(y_data, dtype=bool)

        # 处理 x 范围筛选（如果输入不为空）
        if self.lineEdit_2.text():  # x_min
            x_min = float(self.lineEdit_2.text())
            x_mask &= (x > x_min)  # 与现有条件取交集

        if self.lineEdit_5.text():  # x_max
            x_max = float(self.lineEdit_5.text())
            x_mask &= (x < x_max)

        # 处理 y 范围筛选（如果输入不为空）
        if self.lineEdit_3.text():  # y_min
            y_min = float(self.lineEdit_3.text()) * vrange * 0.01
            y_mask &= (y_data > y_min)

        if self.lineEdit_4.text():  # y_max
            y_max = float(self.lineEdit_4.text()) * vrange * 0.01
            y_mask &= (y_data < y_max)

        # 合并条件并筛选数据
        combined_mask = x_mask & y_mask
        x_filtered = x[combined_mask]
        y_filtered = y_data[combined_mask]

        return x_filtered, y_filtered

    def calculate_decay(self):
                
        try:
            if self.file_path not in self.file_data_cache:
                QMessageBox.warning(self, "错误", f"请选择一份文件")
                return False
            
            if not self.load_file_data(self.file_path):
                QMessageBox.warning(self, "错误", f"无法加载文件数据: {os.path.basename(self.file_path)}")
                return False
                
            data = self.file_data_cache[self.file_path]
        except Exception as e:
            traceback.print_exc()
            return False
                
        try:
            if len(data.shape) == 1 or data.shape[1] < 2:
                raise ValueError("数据列数不足，无法绘制图形")
            
            Origional_x = data[:, 0]  # 第一列作为x轴数据
            Origional_y_data = data[:, 1]  # 第二列作为y轴数据
        except Exception as e:
            print(f"加载文件 {self.file_path} 出错: {str(e)}")
            traceback.print_exc()
            return False
        
        # 创建拟合器
        fitter = MATLABCurveFitter()

        robust_mapping = {
        'Off': RobustMethod.OFF,
        'LAR': RobustMethod.LAR,
        'Bisquare': RobustMethod.BISQUARE
        }

        # 获取当前选择的 robust 方法
        robust_choice = self.comboBox_8.currentText()
        robust_method = robust_mapping.get(robust_choice, RobustMethod.OFF)  # 默认使用 OFF

        algorithm_mapping = {
        'Levenberg-Marquardt': Algorithm.LEVENBERG_MARQUARDT,
        'Trust-Region': Algorithm.TRUST_REGION,
        }

        # 获取当前选择的 robust 方法
        algorithm_choice = self.comboBox_9.currentText()
        algorithm_method = algorithm_mapping.get(algorithm_choice, Algorithm.LEVENBERG_MARQUARDT)  # 默认使用 OFF

        # 设置高级选项
        fitter.set_options(
            Robust=robust_method,
            Algorithm=algorithm_method,
            DiffMinChange = float(self.lineEdit_19.text()),
            DiffMaxChange = float(self.lineEdit_20.text()),
            MaxIter = int(self.lineEdit_21.text()),
            MaxFunEvals = int(self.lineEdit_22.text()),
            TolFun = float(self.lineEdit_23.text()),
            TolX = float(self.lineEdit_24.text())
        )

        # 测试不同模型
        models_to_test = self.comboBox.currentText()

        x, y_data = self.data_range(Origional_x, Origional_y_data)
        result = fitter.fit_curve(x, y_data, models_to_test)


        if models_to_test == '单指数':
            # 获取系数b (对应衰减速率)
            b = result.params[1]  # 参数顺序是[a, b, c]            
            # 计算寿命τ (τ = -1/b)
            lifetime = -1/b if b != 0 else float('inf')
            print(f"文件: {os.path.basename(self.file_path)}")
            print(f"寿命τ: {lifetime}")
            self.lifetime = lifetime
            
            # 在界面上显示
            self.lineEdit.setText(f"{lifetime:.8f}")

        elif models_to_test == '双指数':
            # 获取系数b和d (对应两个衰减速率)
            b = result.params[1]  # 第一个衰减速率
            d = result.params[3]  # 第二个衰减速率            
            # 计算两个寿命
            tau1 = -1/b if b != 0 else float('inf')
            tau2 = -1/d if d != 0 else float('inf')
            
            # 在界面上显示
            self.lineEdit.setText(f"τ1: {tau1:.8f}, τ2: {tau2:.8f}")

        # 绘制图形
        self.figure2.clear()
        ax2 = self.figure2.add_subplot(111)

        # 绘制所有数据点
        ax2.scatter(Origional_x, Origional_y_data, alpha=0.3, color='gray', s=5)
        # 高亮显示用于拟合的数据点
        ax2.scatter(x, y_data, alpha=0.8, color='blue', s=10)
        
        y_pred = fitter._get_model_definition(models_to_test)['function'](x, *result.params)
        ax2.plot(x, y_pred, 'r-', label=f'{models_to_test}拟合', linewidth=2)
        ax2.legend()
        ax2.set_title(f'{models_to_test}模型拟合结果')
        ax2.grid(True, alpha=0.3)    
        self.canvas2.draw()

        # 绘制残差
        self.figure3.clear()
        ax3 = self.figure3.add_subplot(111)
        ax3.plot(x, result.residuals, 'ro-', alpha=0.6, markersize=2)
        ax3.axhline(y=0, color='k', linestyle='--')
        ax3.set_title('残差图')
        ax3.set_xlabel('x')
        ax3.set_ylabel('残差')
        ax3.grid(True, alpha=0.3)
        self.canvas3.draw()
        # self.calculate_lifetime()
    
    def calculate_lifetime(self):
        """使用线性外推方法计算温度"""
        try:
            if self.lifetime == None:
                message="拟合失败！"
            tau_to_interpolate = self.lifetime * 1000

            

            # 检查数据有效性
            if len(self.temperature_calibration) != len(self.lifetime_calibration):
                QMessageBox.warning(self, "警告", "温度-寿命校准数据长度不一致")
                return
            
            if len(self.temperature_calibration) < 2:
                QMessageBox.warning(self, "警告", "温度-寿命校准数据点不足")
                return
            
            import numpy as np
            
            # 对数据进行排序
            sorted_indices = np.argsort(self.lifetime_calibration)
            sorted_lifetime = np.array(self.lifetime_calibration)[sorted_indices]
            sorted_temperature = np.array(self.temperature_calibration)[sorted_indices]
            
            # 确定外推使用的两个边界点
            if tau_to_interpolate <= sorted_lifetime[0]:
                # 使用前两个点进行外推
                x_use = sorted_lifetime[:2]
                y_use = sorted_temperature[:2]
                extrapolation_type = "最小值外推"
            elif tau_to_interpolate >= sorted_lifetime[-1]:
                # 使用最后两个点进行外推
                x_use = sorted_lifetime[-2:]
                y_use = sorted_temperature[-2:]
                extrapolation_type = "最大值外推"
            else:
                # 在范围内，使用numpy插值
                x_use = sorted_lifetime
                y_use = sorted_temperature
                extrapolation_type = "内插"
            
            # 计算线性拟合参数
            if len(x_use) == 2:
                # 两点线性外推/内插
                slope = (y_use[1] - y_use[0]) / (x_use[1] - x_use[0])
                self.temperature_measure = y_use[0] + slope * (tau_to_interpolate - x_use[0])
            else:
                # 使用numpy插值（范围内）
                self.temperature_measure = np.interp(tau_to_interpolate, x_use, y_use)
            
            # 显示结果
            self.textEdit.setText(f'{self.temperature_measure:.2f} °C')
            self.textEdit.setHtml(f'<div align="center"><font color="red" size="7">{self.temperature_measure:.2f} °C</font></div>')
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算寿命温度时出错:{message}")
            traceback.print_exc()
    
    def validate_folder_path(self, path):
        """验证文件夹路径是否有效"""
        return QDir(path).exists()

if __name__ == '__main__':
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    myWin = mainWindow()
    myWin.show()
    sys.exit(app.exec_())