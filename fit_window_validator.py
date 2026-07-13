import concurrent.futures
import csv
import os
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


UNIT_MAPPING = {"s": 1.0, "ms": 1000.0, "μs": 1000000.0, "ns": 1000000000.0}

class FitWindowScanWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, base_task, scan_runner, start_values, end_values, worker_count):
        super().__init__()
        self.base_task = base_task
        self.scan_runner = scan_runner
        self.start_values = [float(v) for v in start_values]
        self.end_values = [float(v) for v in end_values]
        self.worker_count = max(1, int(worker_count))
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def _cancel_check(self):
        return self._cancel_requested

    @QtCore.pyqtSlot()
    def run(self):
        executor = None
        try:
            jobs = [
                (self.base_task, start, end)
                for start in self.start_values
                for end in self.end_values
                if end > start
            ]
            total = len(jobs)
            if total == 0:
                self.failed.emit("没有有效的开始/结束时间组合")
                return

            results = []
            done = 0
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.worker_count)
            futures = [executor.submit(self.scan_runner, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                if self._cancel_requested:
                    for pending in futures:
                        pending.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    self.cancelled.emit()
                    return

                item = future.result()
                if item is not None:
                    results.append(item)
                done += 1
                self.progress.emit(done, total)

            if self._cancel_requested:
                self.cancelled.emit()
                return
            self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if executor is not None and not self._cancel_requested:
                executor.shutdown(wait=True)


class FitWindowValidatorDialog(QtWidgets.QDialog):
    """Scan fitting windows and plot lifetime as a start/end-time surface."""

    def __init__(self, parent, task, scan_runner):
        super().__init__(parent)
        self.task = task
        self.scan_runner = scan_runner
        self.settings = QtCore.QSettings("YourCompany", "PMT_Analysis")
        self._thread = None
        self._worker = None
        self._last_results = []
        self.setWindowTitle("拟合窗验证")
        self.resize(1100, 820)
        self._setup_ui()
        self._set_default_ranges()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.summary_label = QtWidgets.QLabel("设置扫描范围后点击开始。", self)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        form_group = QtWidgets.QGroupBox("扫描设置", self)
        grid = QtWidgets.QGridLayout(form_group)

        labels = ["最小", "最大", "间隔"]
        for col, text in enumerate(labels, start=1):
            grid.addWidget(QtWidgets.QLabel(text, form_group), 0, col)

        grid.addWidget(QtWidgets.QLabel("开始时间", form_group), 1, 0)
        grid.addWidget(QtWidgets.QLabel("结束时间", form_group), 2, 0)

        self.start_min_edit, self.start_min_unit = self._make_time_input()
        self.start_max_edit, self.start_max_unit = self._make_time_input()
        self.start_step_edit, self.start_step_unit = self._make_time_input()
        self.end_min_edit, self.end_min_unit = self._make_time_input()
        self.end_max_edit, self.end_max_unit = self._make_time_input()
        self.end_step_edit, self.end_step_unit = self._make_time_input()

        edits = [
            (self.start_min_edit, self.start_min_unit),
            (self.start_max_edit, self.start_max_unit),
            (self.start_step_edit, self.start_step_unit),
            (self.end_min_edit, self.end_min_unit),
            (self.end_max_edit, self.end_max_unit),
            (self.end_step_edit, self.end_step_unit),
        ]
        for idx, (edit, unit_combo) in enumerate(edits):
            grid.addWidget(self._wrap_time_input(edit, unit_combo), 1 + idx // 3, 1 + idx % 3)

        grid.addWidget(QtWidgets.QLabel("拟合核数量", form_group), 3, 0)
        self.worker_spin = QtWidgets.QSpinBox(form_group)
        self.worker_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.worker_spin.setValue(min(4, max(1, os.cpu_count() or 1)))
        grid.addWidget(self.worker_spin, 3, 1)

        self.start_button = QtWidgets.QPushButton("开始扫描", form_group)
        self.cancel_button = QtWidgets.QPushButton("取消", form_group)
        self.cancel_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_scan)
        self.cancel_button.clicked.connect(self.cancel_scan)
        grid.addWidget(self.start_button, 3, 2)
        grid.addWidget(self.cancel_button, 3, 3)

        layout.addWidget(form_group)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.figure = Figure(figsize=(9, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["开始时间(s)", "结束时间(s)", "寿命(s)", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(190)
        layout.addWidget(self.table)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, self)
        self.export_button = button_box.addButton("导出结果", QtWidgets.QDialogButtonBox.ActionRole)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _make_time_input(self):
        edit = QtWidgets.QLineEdit(self)
        validator = QtGui.QDoubleValidator(edit)
        validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        edit.setValidator(validator)
        unit_combo = QtWidgets.QComboBox(self)
        unit_combo.addItems(["s", "ms", "μs", "ns"])
        return edit, unit_combo

    def _wrap_time_input(self, edit, unit_combo):
        widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(edit, 1)
        layout.addWidget(unit_combo)
        return widget

    def _set_default_ranges(self):
        x = np.asarray(self.task.get("x_for_fitting", []), dtype=float)
        if len(x) == 0:
            return
        if self._restore_saved_ranges():
            return
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        span = max(x_max - x_min, np.finfo(float).eps)
        defaults = self.task.get("fit_window_scan_defaults", {})
        start_unit = defaults.get("start_unit", "s")
        end_unit = defaults.get("end_unit", start_unit)
        start_min = defaults.get("start_min", x_min)
        start_max = defaults.get("start_max", x_max)
        end_min = defaults.get("end_min", x_min)
        end_max = defaults.get("end_max", x_max)
        start_step = defaults.get("start_step", span / 20.0)
        end_step = defaults.get("end_step", span / 20.0)

        self._set_time_edit(self.start_min_edit, self.start_min_unit, start_min, start_unit)
        self._set_time_edit(self.start_max_edit, self.start_max_unit, start_max, start_unit)
        self._set_time_edit(self.start_step_edit, self.start_step_unit, start_step, start_unit)
        self._set_time_edit(self.end_min_edit, self.end_min_unit, end_min, end_unit)
        self._set_time_edit(self.end_max_edit, self.end_max_unit, end_max, end_unit)
        self._set_time_edit(self.end_step_edit, self.end_step_unit, end_step, end_unit)

    def _set_time_edit(self, edit, unit_combo, seconds, unit):
        if unit_combo.findText(unit) < 0:
            unit = "s"
        unit_combo.setCurrentText(unit)
        edit.setText(f"{seconds * UNIT_MAPPING[unit]:.5e}")

    def _restore_saved_ranges(self):
        keys = [
            "start_min", "start_max", "start_step",
            "end_min", "end_max", "end_step",
        ]
        if not all(self.settings.value(f"fit_window_scan/{key}", "") for key in keys):
            return False

        pairs = [
            ("start_min", self.start_min_edit, self.start_min_unit),
            ("start_max", self.start_max_edit, self.start_max_unit),
            ("start_step", self.start_step_edit, self.start_step_unit),
            ("end_min", self.end_min_edit, self.end_min_unit),
            ("end_max", self.end_max_edit, self.end_max_unit),
            ("end_step", self.end_step_edit, self.end_step_unit),
        ]
        for key, edit, unit_combo in pairs:
            edit.setText(str(self.settings.value(f"fit_window_scan/{key}", "")))
            unit = str(self.settings.value(f"fit_window_scan/{key}_unit", unit_combo.currentText()))
            if unit_combo.findText(unit) >= 0:
                unit_combo.setCurrentText(unit)

        workers = self.settings.value("fit_window_scan/workers", "")
        if workers:
            try:
                self.worker_spin.setValue(int(workers))
            except ValueError:
                pass
        return True

    def _save_current_ranges(self):
        pairs = [
            ("start_min", self.start_min_edit, self.start_min_unit),
            ("start_max", self.start_max_edit, self.start_max_unit),
            ("start_step", self.start_step_edit, self.start_step_unit),
            ("end_min", self.end_min_edit, self.end_min_unit),
            ("end_max", self.end_max_edit, self.end_max_unit),
            ("end_step", self.end_step_edit, self.end_step_unit),
        ]
        for key, edit, unit_combo in pairs:
            self.settings.setValue(f"fit_window_scan/{key}", edit.text().strip())
            self.settings.setValue(f"fit_window_scan/{key}_unit", unit_combo.currentText())
        self.settings.setValue("fit_window_scan/workers", self.worker_spin.value())

    def start_scan(self):
        try:
            start_values = self._range_values(
                self.start_min_edit.text(),
                self.start_max_edit.text(),
                self.start_step_edit.text(),
                self.start_min_unit.currentText(),
                self.start_max_unit.currentText(),
                self.start_step_unit.currentText(),
            )
            end_values = self._range_values(
                self.end_min_edit.text(),
                self.end_max_edit.text(),
                self.end_step_edit.text(),
                self.end_min_unit.currentText(),
                self.end_max_unit.currentText(),
                self.end_step_unit.currentText(),
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "参数错误", str(exc))
            return

        valid_count = sum(1 for start in start_values for end in end_values if end > start)
        if valid_count > 100000:
            QtWidgets.QMessageBox.warning(self, "参数错误", "有效拟合窗口超过 100000 个，请增大间隔或缩小范围")
            return
        if valid_count == 0:
            QtWidgets.QMessageBox.warning(self, "参数错误", "没有有效的开始/结束时间组合")
            return

        self._save_current_ranges()
        self._set_busy(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.summary_label.setText("正在扫描拟合窗口...")
        self.figure.clear()
        self.canvas.draw()

        self._thread = QtCore.QThread(self)
        self._worker = FitWindowScanWorker(
            self.task,
            self.scan_runner,
            start_values,
            end_values,
            self.worker_spin.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_thread_refs)
        self._thread.start()

    def cancel_scan(self):
        if self._worker is not None:
            self.summary_label.setText("正在取消...")
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)

    def _range_values(self, min_text, max_text, step_text, min_unit, max_unit, step_unit):
        try:
            start = float(min_text) / UNIT_MAPPING[min_unit]
            stop = float(max_text) / UNIT_MAPPING[max_unit]
            step = float(step_text) / UNIT_MAPPING[step_unit]
        except ValueError:
            raise ValueError("扫描范围和间隔必须是数字")
        if step <= 0:
            raise ValueError("间隔必须大于 0")
        if stop < start:
            raise ValueError("最大值必须大于或等于最小值")
        count = int(np.floor((stop - start) / step)) + 1
        values = start + np.arange(count, dtype=float) * step
        if values[-1] < stop:
            values = np.append(values, stop)
        return values

    def _on_progress(self, done, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.summary_label.setText(f"正在扫描拟合窗口: {done}/{total}")

    def _on_finished(self, results):
        self._last_results = list(results)
        success_count = sum(1 for item in self._last_results if item.get("success"))
        self.summary_label.setText(
            f"扫描完成：成功 {success_count}/{len(self._last_results)} 个窗口。"
        )
        self._plot_surface(self._last_results)
        self._fill_table(self._last_results)
        self.export_button.setEnabled(bool(self._last_results))
        self._set_busy(False)

    def _on_failed(self, message):
        self.summary_label.setText("扫描失败。")
        self._set_busy(False)
        QtWidgets.QMessageBox.warning(self, "拟合窗验证失败", message)

    def _on_cancelled(self):
        self.summary_label.setText("扫描已取消。")
        self._set_busy(False)

    def _clear_thread_refs(self):
        self._thread = None
        self._worker = None

    def _set_busy(self, busy):
        self.start_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.worker_spin.setEnabled(not busy)
        self.export_button.setEnabled((not busy) and bool(self._last_results))
        for edit in [
            self.start_min_edit, self.start_max_edit, self.start_step_edit,
            self.end_min_edit, self.end_max_edit, self.end_step_edit,
        ]:
            edit.setEnabled(not busy)
        for combo in [
            self.start_min_unit, self.start_max_unit, self.start_step_unit,
            self.end_min_unit, self.end_max_unit, self.end_step_unit,
        ]:
            combo.setEnabled(not busy)

    def _plot_surface(self, results):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")

        starts = sorted({item["start"] for item in results})
        ends = sorted({item["end"] for item in results})
        z = np.full((len(ends), len(starts)), np.nan, dtype=float)
        start_index = {value: idx for idx, value in enumerate(starts)}
        end_index = {value: idx for idx, value in enumerate(ends)}

        for item in results:
            z[end_index[item["end"]], start_index[item["start"]]] = item["lifetime"]

        x_grid, y_grid = np.meshgrid(starts, ends)
        if np.isfinite(z).any():
            ax.plot_surface(x_grid, y_grid, z, cmap="viridis", edgecolor="none", alpha=0.9)
            ax.scatter(x_grid[np.isfinite(z)], y_grid[np.isfinite(z)], z[np.isfinite(z)], s=8, color="black")
        ax.set_xlabel("开始时间 (s)")
        ax.set_ylabel("结束时间 (s)")
        ax.set_zlabel("寿命 (s)")
        ax.set_title("拟合窗寿命曲面")
        self.figure.tight_layout()
        self.canvas.draw()

    def _fill_table(self, results):
        visible_results = sorted(results, key=lambda item: (item["start"], item["end"]))
        self.table.setRowCount(len(visible_results))
        for row, item in enumerate(visible_results):
            lifetime = item.get("lifetime")
            lifetime_text = f"{lifetime:.5e}" if lifetime is not None and np.isfinite(lifetime) else "N/A"
            values = [
                f"{item['start']:.5e}",
                f"{item['end']:.5e}",
                lifetime_text,
                "成功" if item.get("success") else item.get("message", "失败"),
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(value)))

    def export_results(self):
        if not self._last_results:
            QtWidgets.QMessageBox.warning(self, "导出结果", "没有可导出的扫描结果")
            return

        initial_path = self._default_export_path()
        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出拟合窗验证结果",
            initial_path,
            "CSV Files (*.csv);;Text Files (*.txt)"
        )
        if not file_path:
            return

        if selected_filter.startswith("CSV") and not file_path.lower().endswith(".csv"):
            file_path += ".csv"
        elif selected_filter.startswith("Text") and not file_path.lower().endswith(".txt"):
            file_path += ".txt"

        try:
            delimiter = "," if file_path.lower().endswith(".csv") else "\t"
            with open(file_path, "w", newline="", encoding="utf-8-sig") as output_file:
                writer = csv.writer(output_file, delimiter=delimiter)
                param_names = next(
                    (item.get("param_names", []) for item in self._last_results if item.get("param_names")),
                    [],
                )
                writer.writerow(
                    ["开始时间(s)", "结束时间(s)", "文件名", "模型类型"]
                    + [f"系数_{name}" for name in param_names]
                    + ["SSE", "R平方", "调整R平方", "RMSE", "拟合成功", "平均寿命(s)"]
                )
                for item in sorted(self._last_results, key=lambda row: (row["start"], row["end"])):
                    lifetime = item.get("lifetime")
                    params = list(item.get("params", []))
                    params.extend([np.nan] * (len(param_names) - len(params)))
                    writer.writerow([
                        f"{item['start']:.12e}",
                        f"{item['end']:.12e}",
                        item.get("file_name", ""),
                        item.get("model_type", ""),
                        *[f"{value:.5e}" if np.isfinite(value) else "" for value in params],
                        f"{item.get('sse', np.nan):.5e}",
                        f"{item.get('rsquare', np.nan):.5e}",
                        f"{item.get('adjrsquare', np.nan):.5e}",
                        f"{item.get('rmse', np.nan):.5e}",
                        "是" if item.get("success") else "否",
                        f"{lifetime:.5e}" if lifetime is not None and np.isfinite(lifetime) else "",
                    ])
            QtWidgets.QMessageBox.information(self, "导出完成", f"结果已导出到:\n{file_path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导出失败", f"导出结果时出错:\n{exc}")

    def _default_export_path(self):
        file_path = self.task.get("file_path", "")
        base_name = os.path.splitext(os.path.basename(file_path))[0] if file_path else "fit_window_scan"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"{base_name}_fit_window_scan_{timestamp}.csv"

        defaults = self.task.get("fit_window_scan_defaults", {})
        output_dir = defaults.get("output_dir", "")
        if output_dir and os.path.isdir(output_dir):
            return os.path.join(output_dir, file_name)

        if file_path:
            return os.path.join(os.path.dirname(file_path), file_name)
        return file_name

    def closeEvent(self, event):
        self._save_current_ranges()
        if self._thread is not None and self._thread.isRunning():
            self.summary_label.setText("扫描仍在运行，正在取消...")
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            event.ignore()
            return
        super().closeEvent(event)
