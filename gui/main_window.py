# gui/main_window.py

import sys
import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QLineEdit, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QVBoxLayout, QHBoxLayout, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class HDRProcessingThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._is_running = True

    def run(self):
        import subprocess
        # 构建命令行参数
        cmd = [
            sys.executable, 'src/main.py',
            '-i', self.args['input'],
            '-f', self.args['feature_detector'],
            '-t', self.args['tone_mapping'],
            '--gamma', str(self.args['gamma']),
            '--saturation_scale', str(self.args['saturation_scale']),
            '--hue_shift', str(self.args['hue_shift']),
        ]
        # 启动子进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        for line in process.stdout:
            if not self._is_running:
                process.terminate()
                break
            self.log_signal.emit(line)
        process.wait()
        self.finished_signal.emit()

    def stop(self):
        self._is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDR多帧合成处理系统")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()
        self.thread = None

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 输入选择
        input_label = QLabel("输入文件夹:")
        self.input_line = QLineEdit()
        input_browse = QPushButton("浏览")
        input_browse.clicked.connect(self.browse_input)

        input_layout = QHBoxLayout()
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_line)
        input_layout.addWidget(input_browse)

        # 参数设置
        feature_label = QLabel("特征检测算法:")
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(["SIFT", "ORB"])

        tone_label = QLabel("色调映射算法:")
        self.tone_combo = QComboBox()
        self.tone_combo.addItems(["Reinhard", "Drago", "Durand"])

        gamma_label = QLabel("Gamma值:")
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 5.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(0.2)

        saturation_label = QLabel("饱和度缩放比例:")
        self.saturation_spin = QDoubleSpinBox()
        self.saturation_spin.setRange(0.0, 5.0)
        self.saturation_spin.setSingleStep(0.1)
        self.saturation_spin.setValue(1.0)

        hue_label = QLabel("色调偏移量 (度):")
        self.hue_spin = QDoubleSpinBox()
        self.hue_spin.setRange(-180.0, 180.0)
        self.hue_spin.setSingleStep(1.0)
        self.hue_spin.setValue(0.0)

        params_layout = QHBoxLayout()
        params_layout.addWidget(feature_label)
        params_layout.addWidget(self.feature_combo)
        params_layout.addWidget(tone_label)
        params_layout.addWidget(self.tone_combo)
        params_layout.addWidget(gamma_label)
        params_layout.addWidget(self.gamma_spin)
        params_layout.addWidget(saturation_label)
        params_layout.addWidget(self.saturation_spin)
        params_layout.addWidget(hue_label)
        params_layout.addWidget(self.hue_spin)

        # 输出选择
        output_label = QLabel("输出文件夹:")
        self.output_line = QLineEdit()
        output_browse = QPushButton("浏览")
        output_browse.clicked.connect(self.browse_output)

        output_layout = QHBoxLayout()
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_line)
        output_layout.addWidget(output_browse)

        # 控制按钮
        self.start_button = QPushButton("开始处理")
        self.start_button.clicked.connect(self.start_processing)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.cancel_button)

        # 日志显示
        log_label = QLabel("日志:")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 无限进度条

        # 总布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(params_layout)
        main_layout.addLayout(output_layout)
        main_layout.addLayout(buttons_layout)
        main_layout.addWidget(log_label)
        main_layout.addWidget(self.log_text)
        main_layout.addWidget(self.progress_bar)

        central_widget.setLayout(main_layout)

    def browse_input(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if directory:
            self.input_line.setText(directory)
            # 自动设置输出文件夹为输入文件夹下的 "output"
            output_dir = os.path.join(directory, "output")
            self.output_line.setText(output_dir)

    def browse_output(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if directory:
            self.output_line.setText(directory)

    def start_processing(self):
        input_dir = self.input_line.text().strip()
        output_dir = self.output_line.text().strip()

        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.critical(self, "错误", "请输入有效的输入文件夹路径。")
            return

        if not output_dir:
            QMessageBox.critical(self, "错误", "请输入有效的输出文件夹路径。")
            return

        args = {
            'input': input_dir,
            'feature_detector': self.feature_combo.currentText(),
            'tone_mapping': self.tone_combo.currentText(),
            'gamma': self.gamma_spin.value(),
            'saturation_scale': self.saturation_spin.value(),
            'hue_shift': self.hue_spin.value(),
        }

        # 创建输出文件夹
        os.makedirs(output_dir, exist_ok=True)

        # 禁用开始按钮，启用取消按钮
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.log_text.clear()
        self.progress_bar.setRange(0, 0)  # 开始时设置为无限进度

        # 启动处理线程
        self.thread = HDRProcessingThread(args)
        self.thread.log_signal.connect(self.update_log)
        self.thread.finished_signal.connect(self.processing_finished)
        self.thread.start()

    def cancel_processing(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.log_text.append("用户取消了处理。")
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

    def update_log(self, message):
        self.log_text.append(message)

    def processing_finished(self):
        self.log_text.append("处理完成。")
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "完成", "HDR多帧合成处理已完成。")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
