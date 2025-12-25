import sys
import os
import time
import gc
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QStackedWidget,
    QGroupBox, QProgressBar, QMessageBox, QComboBox,
    QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont
import torch

# 导入模型（确保model.py在当前目录）
try:
    from model import FunASRNano
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"无法导入模型: {e}")
    MODEL_AVAILABLE = False


class WorkerThread(QThread):
    """工作线程，用于加载模型避免界面卡顿"""
    progress_signal = Signal(str, int, int, str)  # stage, value, total, message
    finished_signal = Signal(object, object, bool, str)  # model, kwargs, success, message
    
    def __init__(self, model_dir, device):
        super().__init__()
        self.model_dir = model_dir
        self.device = device
        
    def run(self):
        try:
            self.progress_signal.emit("检查模型路径", 10, 100, "正在检查模型路径...")
            model_dir = os.path.abspath(self.model_dir)
            
            if not os.path.isdir(model_dir):
                self.finished_signal.emit(None, None, False, f"模型目录不存在: {model_dir}")
                return
                
            self.progress_signal.emit("加载模型", 30, 100, "正在加载模型...")
            if not MODEL_AVAILABLE:
                self.finished_signal.emit(None, None, False, "模型模块无法导入，请检查model.py")
                return
                
            m, kwargs = FunASRNano.from_pretrained(model=model_dir, device=self.device)
            m.eval()
            
            # 加载完成后立即进行垃圾回收
            self.progress_signal.emit("清理内存", 95, 100, "正在清理内存...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            self.progress_signal.emit("完成", 100, 100, "模型加载成功！")
            self.finished_signal.emit(m, kwargs, True, "模型加载成功！")
            
        except Exception as e:
            self.finished_signal.emit(None, None, False, f"模型加载失败: {str(e)}")


class TranscriptionThread(QThread):
    """转录工作线程"""
    progress_signal = Signal(str, int, int, str)  # stage, value, total, message
    result_signal = Signal(str, bool, str)  # result, success, message
    
    def __init__(self, model, kwargs, audio_path, output_path=None):
        super().__init__()
        self.model = model
        self.kwargs = kwargs
        self.audio_path = audio_path
        self.output_path = output_path
        self.start_time = None
        
    def run(self):
        try:
            self.start_time = time.time()
            
            # 设置进度回调
            self.model.set_progress_callback(self._progress_callback)
            
            self.progress_signal.emit("准备音频", 5, 100, "正在准备音频文件...")
            
            # 检查音频文件是否存在
            if not os.path.isfile(self.audio_path):
                self.result_signal.emit("", False, f"音频文件不存在: {self.audio_path}")
                return
                
            # 执行推理
            res = self.model.inference(data_in=[self.audio_path], **self.kwargs)
            
            # 解析结果
            if isinstance(res, tuple) and len(res) > 0 and isinstance(res[0], list) and len(res[0]) > 0:
                text = res[0][0].get("text", "未找到文本")
                text = text.strip()
                
                # 计算耗时
                elapsed_time = time.time() - self.start_time
                
                # 如果指定了输出文件，保存结果
                if self.output_path:
                    try:
                        with open(self.output_path, 'w', encoding='utf-8') as f:
                            f.write(f"音频文件: {self.audio_path}\n")
                            f.write(f"识别结果:\n{text}\n")
                            f.write(f"处理时间: {elapsed_time:.2f}秒\n")
                        self.result_signal.emit(text, True, f"转录完成！耗时{elapsed_time:.1f}秒，结果已保存到: {self.output_path}")
                    except Exception as e:
                        self.result_signal.emit(text, True, f"转录完成！耗时{elapsed_time:.1f}秒，但保存文件失败: {str(e)}")
                else:
                    self.result_signal.emit(text, True, f"转录完成！耗时{elapsed_time:.1f}秒")
            else:
                self.result_signal.emit("", False, "无法解析识别结果")
                
        except Exception as e:
            self.result_signal.emit("", False, f"转录过程中发生错误: {str(e)}")
            
    def _progress_callback(self, stage, value, total, message):
        """模型进度回调"""
        self.progress_signal.emit(stage, value, total, message)


class ModelLoadPage(QWidget):
    """模型加载页面"""
    model_loaded = Signal(object, object, str)  # model, kwargs, model_dir
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # 标题
        title = QLabel("加载模型")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 当前模型状态
        self.model_status_group = QGroupBox("当前模型状态")
        self.model_status_group.setMaximumHeight(80)
        model_status_layout = QVBoxLayout()
        self.current_model_label = QLabel("尚未加载任何模型")
        self.current_model_label.setStyleSheet("padding: 8px; background: #f5f5f5; border-radius: 3px;")
        model_status_layout.addWidget(self.current_model_label)
        
        self.model_status_group.setLayout(model_status_layout)
        self.model_status_group.setVisible(True)
        layout.addWidget(self.model_status_group)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # 模型选择区域
        model_group = QGroupBox("加载模型")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(8)
        
        # 模型路径选择
        path_layout = QHBoxLayout()
        self.path_label = QLabel("未选择模型目录")
        self.path_label.setStyleSheet("border: 1px solid #ddd; padding: 6px; background: white;")
        self.path_label.setMinimumHeight(35)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_model_dir)
        browse_btn.setMaximumWidth(80)
        
        path_layout.addWidget(self.path_label)
        path_layout.addWidget(browse_btn)
        model_layout.addLayout(path_layout)
        
        # 设备选择
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("设备:"))
        
        self.device_combo = QComboBox()
        self.device_combo.addItem("自动选择 (优先使用GPU)")
        self.device_combo.addItem("CPU")
        self.device_combo.addItem("GPU (CUDA)")
        self.device_combo.setMaximumWidth(200)
        
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        model_layout.addLayout(device_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 进度显示
        progress_group = QGroupBox("加载进度")
        progress_layout = QVBoxLayout()
        
        self.stage_label = QLabel("准备加载")
        self.stage_label.setStyleSheet("font-weight: bold; color: #333;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        
        self.time_label = QLabel("预计剩余时间: --")
        self.time_label.setStyleSheet("color: #666; font-size: 12px;")
        
        self.status_label = QLabel("等待操作...")
        self.status_label.setStyleSheet("color: #666; padding: 5px 0;")
        
        progress_layout.addWidget(self.stage_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.time_label)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 加载按钮
        self.load_btn = QPushButton("加载模型")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.load_btn.clicked.connect(self.load_model)
        self.load_btn.setMinimumHeight(40)
        layout.addWidget(self.load_btn)
        
        # 底部信息
        info_label = QLabel("提示：加载新模型时会自动卸载当前模型")
        info_label.setStyleSheet("color: #666; font-size: 11px; padding: 8px;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def browse_model_dir(self):
        """浏览选择模型目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择模型目录", 
            str(Path.home()),
            QFileDialog.ShowDirsOnly
        )
        if dir_path:
            self.path_label.setText(dir_path)
            self.status_label.setText(f"已选择: {os.path.basename(dir_path)}")
            
    def update_model_status(self, model_name, model_path):
        """更新模型状态显示"""
        if model_name and model_path:
            self.current_model_label.setText(f"✓ 已加载: {model_name}")
        else:
            self.current_model_label.setText("尚未加载任何模型")
            
    def load_model(self):
        """加载模型"""
        model_dir = self.path_label.text()
        
        if model_dir == "未选择模型目录" or not os.path.isdir(model_dir):
            QMessageBox.warning(self, "警告", "请先选择有效的模型目录！")
            return
            
        # 获取设备设置
        device_choice = self.device_combo.currentText()
        if device_choice == "自动选择 (优先使用GPU)":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device_choice == "GPU (CUDA)":
            if not torch.cuda.is_available():
                QMessageBox.warning(self, "警告", "CUDA不可用，请检查GPU驱动或选择其他设备！")
                return
            device = "cuda:0"
        else:
            device = "cpu"
            
        # 先卸载当前模型（如果有）
        if self.main_window.model is not None:
            self.status_label.setText("正在卸载当前模型...")
            QApplication.processEvents()  # 处理UI更新
            self.main_window.unload_model()
            
        # 禁用按钮，显示进度
        self.load_btn.setEnabled(False)
        self.stage_label.setText("正在加载模型...")
        self.progress_bar.setValue(0)
        self.start_time = time.time()
        
        # 启动定时器更新剩余时间
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_estimated_time)
        self.timer.start(1000)  # 每秒更新一次
        
        # 在工作线程中加载模型
        self.worker = WorkerThread(model_dir, device)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.model_loaded_callback)
        self.worker.start()
        
    def update_progress(self, stage, value, total, message):
        """更新进度信息"""
        self.stage_label.setText(stage)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
    def update_estimated_time(self):
        """更新预计剩余时间"""
        if hasattr(self, 'start_time') and self.progress_bar.value() > 0:
            elapsed = time.time() - self.start_time
            progress = self.progress_bar.value()
            total = self.progress_bar.maximum()
            
            if progress > 0:
                remaining = (elapsed / progress) * (total - progress)
                self.time_label.setText(f"预计剩余时间: {remaining:.0f}秒")
            else:
                self.time_label.setText("预计剩余时间: --")
        
    def model_loaded_callback(self, model, kwargs, success, message):
        """模型加载完成回调"""
        # 停止定时器
        if hasattr(self, 'timer'):
            self.timer.stop()
            
        self.load_btn.setEnabled(True)
        
        if success:
            model_dir = self.path_label.text()
            model_name = os.path.basename(model_dir)
            
            # 更新当前模型状态显示
            self.update_model_status(model_name, model_dir)
            
            self.status_label.setText("模型加载成功！")
            self.time_label.setText("加载完成！")
            
            # 发送信号通知主窗口
            self.model_loaded.emit(model, kwargs, model_dir)
            
            # 切换到转录页面
            self.main_window.switch_to_transcription_page()
        else:
            self.status_label.setText(f"加载失败: {message}")
            self.time_label.setText("加载失败")
            QMessageBox.critical(self, "错误", message)


class TranscriptionPage(QWidget):
    """音频转录页面"""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # 标题
        title = QLabel("音频转录")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 当前模型状态
        self.model_status_display = QLabel("未加载模型")
        self.model_status_display.setStyleSheet("""
            padding: 8px; 
            background: #fff3cd; 
            border: 1px solid #ffeaa7; 
            border-radius: 4px; 
            color: #856404;
            font-size: 12px;
        """)
        layout.addWidget(self.model_status_display)
        
        # 音频文件选择
        audio_group = QGroupBox("音频文件")
        audio_layout = QVBoxLayout()
        
        audio_select_layout = QHBoxLayout()
        self.audio_label = QLabel("未选择音频文件")
        self.audio_label.setStyleSheet("border: 1px solid #ddd; padding: 6px; background: white;")
        self.audio_label.setMinimumHeight(35)
        
        audio_browse_btn = QPushButton("浏览...")
        audio_browse_btn.clicked.connect(self.browse_audio_file)
        audio_browse_btn.setMaximumWidth(80)
        
        audio_select_layout.addWidget(self.audio_label)
        audio_select_layout.addWidget(audio_browse_btn)
        audio_layout.addLayout(audio_select_layout)
        
        # 支持格式提示
        format_label = QLabel("支持格式: WAV, MP3, FLAC, M4A, OGG 等")
        format_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        audio_layout.addWidget(format_label)
        
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        
        # 输出设置
        output_group = QGroupBox("输出设置 (可选)")
        output_layout = QVBoxLayout()
        
        # 输出文件选择
        output_select_layout = QHBoxLayout()
        self.output_label = QLabel("未选择输出文件")
        self.output_label.setStyleSheet("border: 1px solid #ddd; padding: 6px; background: white;")
        self.output_label.setMinimumHeight(35)
        
        output_browse_btn = QPushButton("浏览...")
        output_browse_btn.clicked.connect(self.browse_output_file)
        output_browse_btn.setMaximumWidth(80)
        
        output_select_layout.addWidget(self.output_label)
        output_select_layout.addWidget(output_browse_btn)
        output_layout.addLayout(output_select_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 进度显示
        progress_group = QGroupBox("转录进度")
        progress_layout = QVBoxLayout()
        
        self.stage_label = QLabel("准备转录")
        self.stage_label.setStyleSheet("font-weight: bold; color: #333;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        
        self.time_label = QLabel("预计剩余时间: --")
        self.time_label.setStyleSheet("color: #666; font-size: 12px;")
        
        self.status_label = QLabel("等待开始转录...")
        self.status_label.setStyleSheet("color: #666; padding: 5px 0;")
        
        progress_layout.addWidget(self.stage_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.time_label)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 转录结果
        result_group = QGroupBox("转录结果")
        result_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(150)
        result_layout.addWidget(self.result_text)
        
        # 结果操作按钮
        result_buttons_layout = QHBoxLayout()
        
        self.copy_btn = QPushButton("复制结果")
        self.copy_btn.clicked.connect(self.copy_result)
        self.copy_btn.setEnabled(False)
        
        self.clear_result_btn = QPushButton("清空结果")
        self.clear_result_btn.clicked.connect(self.clear_result)
        
        result_buttons_layout.addWidget(self.copy_btn)
        result_buttons_layout.addStretch()
        result_buttons_layout.addWidget(self.clear_result_btn)
        result_layout.addLayout(result_buttons_layout)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # 转录按钮
        self.transcribe_btn = QPushButton("开始转录")
        self.transcribe_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.transcribe_btn.setMinimumHeight(40)
        layout.addWidget(self.transcribe_btn)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def update_model_status(self, model_name):
        """更新模型状态显示"""
        if model_name:
            self.model_status_display.setText(f"✓ 已加载模型: {model_name}")
            self.model_status_display.setStyleSheet("""
                padding: 8px; 
                background: #d4edda; 
                border: 1px solid #c3e6cb; 
                border-radius: 4px; 
                color: #155724;
                font-size: 12px;
            """)
            self.transcribe_btn.setEnabled(True)
        else:
            self.model_status_display.setText("未加载模型")
            self.model_status_display.setStyleSheet("""
                padding: 8px; 
                background: #fff3cd; 
                border: 1px solid #ffeaa7; 
                border-radius: 4px; 
                color: #856404;
                font-size: 12px;
            """)
            self.transcribe_btn.setEnabled(False)
            
    def browse_audio_file(self):
        """浏览选择音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件",
            str(Path.home()),
            "音频文件 (*.wav *.mp3 *.flac *.m4a *.ogg);;所有文件 (*.*)"
        )
        if file_path:
            self.audio_label.setText(file_path)
            self.status_label.setText(f"已选择音频文件: {os.path.basename(file_path)}")
            
    def browse_output_file(self):
        """浏览选择输出文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择输出文件",
            str(Path.home()),
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if file_path:
            self.output_label.setText(file_path)
            
    def start_transcription(self):
        """开始转录"""
        audio_path = self.audio_label.text()
        
        if audio_path == "未选择音频文件" or not os.path.isfile(audio_path):
            QMessageBox.warning(self, "警告", "请先选择有效的音频文件！")
            return
            
        if self.main_window.model is None:
            QMessageBox.warning(self, "警告", "模型未加载，请先在'加载模型'页面加载模型！")
            return
            
        # 获取输出文件路径（可选）
        output_path = self.output_label.text()
        if output_path == "未选择输出文件":
            output_path = None
            
        # 禁用按钮，显示进度
        self.transcribe_btn.setEnabled(False)
        self.stage_label.setText("开始转录...")
        self.progress_bar.setValue(0)
        self.start_time = time.time()
        
        # 启动定时器更新剩余时间
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_estimated_time)
        self.timer.start(1000)  # 每秒更新一次
        
        # 在工作线程中执行转录
        self.worker = TranscriptionThread(
            self.main_window.model,
            self.main_window.model_kwargs,
            audio_path,
            output_path
        )
        self.worker.progress_signal.connect(self.update_transcription_progress)
        self.worker.result_signal.connect(self.transcription_finished)
        self.worker.start()
        
    def update_transcription_progress(self, stage, value, total, message):
        """更新转录进度"""
        self.stage_label.setText(stage)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
    def update_estimated_time(self):
        """更新预计剩余时间"""
        if hasattr(self, 'start_time') and self.progress_bar.value() > 0:
            elapsed = time.time() - self.start_time
            progress = self.progress_bar.value()
            total = self.progress_bar.maximum()
            
            if progress > 0:
                remaining = (elapsed / progress) * (total - progress)
                self.time_label.setText(f"预计剩余时间: {remaining:.0f}秒")
            else:
                self.time_label.setText("预计剩余时间: --")
        
    def transcription_finished(self, result, success, message):
        """转录完成回调"""
        # 停止定时器
        if hasattr(self, 'timer'):
            self.timer.stop()
            
        self.transcribe_btn.setEnabled(True)
        
        if success:
            self.status_label.setText(message)
            self.result_text.setText(result)
            self.copy_btn.setEnabled(True)
            self.time_label.setText("转录完成！")
            
            # 转录完成后进行垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            self.status_label.setText(f"转录失败: {message}")
            self.time_label.setText("转录失败")
            QMessageBox.critical(self, "错误", message)
            
    def copy_result(self):
        """复制结果到剪贴板"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.result_text.toPlainText())
        self.status_label.setText("结果已复制到剪贴板")
        
    def clear_result(self):
        """清空结果"""
        self.result_text.clear()
        self.copy_btn.setEnabled(False)
        self.status_label.setText("结果已清空")


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_kwargs = None
        self.current_model_name = ""
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("FunASR-Nano 语音识别工具")
        self.setGeometry(100, 100, 700, 650)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 标题栏
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: #2c3e50;")
        title_layout = QHBoxLayout(title_frame)
        
        title_label = QLabel("FunASR-Nano 语音识别工具")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: white; padding: 12px;")
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # 导航按钮
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(5)
        
        self.model_btn = QPushButton("加载模型")
        self.model_btn.setCheckable(True)
        self.model_btn.setChecked(True)
        self.model_btn.clicked.connect(lambda: self.switch_to_model_page())
        
        self.transcribe_btn = QPushButton("音频转录")
        self.transcribe_btn.setCheckable(True)
        self.transcribe_btn.clicked.connect(lambda: self.switch_to_transcription_page())
        
        # 按钮组样式
        button_style = """
            QPushButton {
                background-color: #34495e;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 3px;
                font-size: 13px;
            }
            QPushButton:checked {
                background-color: #1abc9c;
            }
            QPushButton:hover {
                background-color: #3d566e;
            }
        """
        
        self.model_btn.setStyleSheet(button_style)
        self.transcribe_btn.setStyleSheet(button_style)
        
        nav_layout.addWidget(self.model_btn)
        nav_layout.addWidget(self.transcribe_btn)
        
        title_layout.addLayout(nav_layout)
        
        main_layout.addWidget(title_frame)
        
        # 创建堆叠窗口
        self.stacked_widget = QStackedWidget()
        
        # 创建页面
        self.model_page = ModelLoadPage(self)
        self.transcription_page = TranscriptionPage(self)
        
        # 连接模型加载信号
        self.model_page.model_loaded.connect(self.on_model_loaded)
        
        # 添加页面到堆叠窗口
        self.stacked_widget.addWidget(self.model_page)
        self.stacked_widget.addWidget(self.transcription_page)
        
        main_layout.addWidget(self.stacked_widget)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
    def switch_to_model_page(self):
        """切换到模型加载页面"""
        self.model_btn.setChecked(True)
        self.transcribe_btn.setChecked(False)
        self.stacked_widget.setCurrentIndex(0)
        self.statusBar().showMessage("模型加载页面")
        
    def switch_to_transcription_page(self):
        """切换到音频转录页面"""
        self.transcribe_btn.setChecked(True)
        self.model_btn.setChecked(False)
        self.stacked_widget.setCurrentIndex(1)
        
        # 更新转录页面的模型状态显示
        self.transcription_page.update_model_status(self.current_model_name)
        
        # 检查是否有模型加载
        if self.model is None:
            self.transcription_page.status_label.setText("请先在'加载模型'页面加载模型")
            self.statusBar().showMessage("未加载模型，请先加载模型")
        else:
            self.statusBar().showMessage("音频转录页面 - 已加载模型")
            
    def unload_model(self):
        """彻底卸载当前模型，释放所有内存"""
        if self.model is not None:
            try:
                print("开始卸载模型...")
                
                # 1. 清除进度回调
                if hasattr(self.model, 'set_progress_callback'):
                    self.model.set_progress_callback(None)
                
                # 2. 获取所有子模块引用
                submodules = []
                if hasattr(self.model, 'audio_encoder'):
                    submodules.append(self.model.audio_encoder)
                if hasattr(self.model, 'audio_adaptor'):
                    submodules.append(self.model.audio_adaptor)
                if hasattr(self.model, 'llm'):
                    submodules.append(self.model.llm)
                
                # 3. 将模型移到CPU（如果它在GPU上）
                if next(self.model.parameters()).is_cuda:
                    self.model.to('cpu')
                
                # 4. 删除所有子模块的属性
                if hasattr(self.model, 'audio_encoder'):
                    self.model.audio_encoder = None
                if hasattr(self.model, 'audio_adaptor'):
                    self.model.audio_adaptor = None
                if hasattr(self.model, 'llm'):
                    self.model.llm = None
                
                # 5. 删除模型对象
                del self.model
                self.model = None
                
                # 6. 删除所有子模块
                for module in submodules:
                    if module is not None:
                        # 递归删除子模块
                        self._recursive_delete(module)
                
                # 7. 删除kwargs
                if self.model_kwargs is not None:
                    # 清理kwargs中可能的张量
                    for key, value in list(self.model_kwargs.items()):
                        if isinstance(value, torch.Tensor):
                            del value
                    self.model_kwargs.clear()
                    self.model_kwargs = None
                
                # 8. 强制垃圾回收
                gc.collect()
                
                # 9. 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                # 10. 再次垃圾回收确保清理
                gc.collect()
                
                # 更新状态
                self.current_model_name = ""
                
                # 更新页面显示
                self.model_page.update_model_status("", "")
                self.transcription_page.update_model_status("")
                
                print("模型卸载完成")
                
            except Exception as e:
                print(f"卸载模型时出错: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _recursive_delete(self, module):
        """递归删除模块及其子模块"""
        if module is None:
            return
            
        try:
            # 如果是nn.Module，先删除其子模块
            if isinstance(module, torch.nn.Module):
                # 获取所有子模块名称
                children_names = list(module._modules.keys())
                for name in children_names:
                    child = module._modules.get(name, None)
                    if child is not None:
                        # 递归删除子模块
                        self._recursive_delete(child)
                        # 从父模块中移除
                        module._modules.pop(name, None)
                
                # 删除参数
                for param_name, param in list(module._parameters.items()):
                    if param is not None:
                        del param
                        module._parameters.pop(param_name, None)
                
                # 删除缓冲区
                for buffer_name, buffer in list(module._buffers.items()):
                    if buffer is not None:
                        del buffer
                        module._buffers.pop(buffer_name, None)
            
            # 删除模块本身
            del module
            
        except Exception as e:
            print(f"删除模块时出错: {str(e)}")
                
    def on_model_loaded(self, model, kwargs, model_dir):
        """处理模型加载完成信号"""
        # 先卸载当前模型（如果有）
        if self.model is not None:
            self.unload_model()
            
        self.model = model
        self.model_kwargs = kwargs
        self.current_model_name = os.path.basename(model_dir)
        
        if model:
            # 加载完成后进行垃圾回收，清理加载过程中的临时变量
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.statusBar().showMessage(f"模型加载成功: {self.current_model_name}")
            
            # 更新两个页面的模型状态显示
            self.model_page.update_model_status(self.current_model_name, model_dir)
            self.transcription_page.update_model_status(self.current_model_name)
            
            # 切换到转录页面
            self.switch_to_transcription_page()
        else:
            self.statusBar().showMessage("模型加载失败")
            
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 彻底卸载模型
        self.unload_model()
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    # 设置应用字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    # 检查依赖
    if not MODEL_AVAILABLE:
        reply = QMessageBox.question(
            None, "缺少模型模块",
            "无法导入model.py模块。\n\n"
            "请确保model.py文件在当前目录中，并且所有依赖已安装。\n\n"
            "是否要继续运行？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            sys.exit(1)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
