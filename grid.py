import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QGroupBox, QGridLayout, QMessageBox,
                             QTabWidget, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class MSFEGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MSFE Surface Generator for Zemax Grid Sag")
        self.setGeometry(100, 100, 1400, 900)
        
        # Основной виджет
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Левая панель - параметры
        left_panel = self.create_parameters_panel()
        layout.addWidget(left_panel, stretch=1)
        
        # Правая панель - графики
        right_panel = self.create_plots_panel()
        layout.addWidget(right_panel, stretch=3)
        
        # Инициализация данных
        self.X = None
        self.Y = None
        self.Z = None
        self.Z_base = None
        
    def create_parameters_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Группа: Базовая форма
        base_group = QGroupBox("Базовая форма (Grid Sag parameters)")
        base_layout = QGridLayout()
        
        self.radius_input = self.create_parameter_row(
            base_layout, 0, "Радиус (мм):", "100"
        )
        self.conic_input = self.create_parameter_row(
            base_layout, 1, "Коническая постоянная:", "-1.0"
        )
        
        base_layout.addWidget(QLabel("(Базовая форма будет задана\nчерез Radius и Conic в Zemax)"), 2, 0, 1, 2)
        
        base_group.setLayout(base_layout)
        layout.addWidget(base_group)
        
        # Группа: Геометрия
        geometry_group = QGroupBox("Геометрия Grid Sag")
        geometry_layout = QGridLayout()
        
        self.aperture_input = self.create_parameter_row(
            geometry_layout, 0, "Диаметр апертуры (мм):", "100"
        )
        self.grid_size_input = self.create_parameter_row(
            geometry_layout, 1, "Размер сетки (nx=ny):", "256"
        )
        
        geometry_group.setLayout(geometry_layout)
        layout.addWidget(geometry_group)
        
        # Группа: Ступенчатые ошибки
        step_group = QGroupBox("Параметры ступенек (осесимметричные)")
        step_layout = QGridLayout()
        
        step_layout.addWidget(QLabel("Тип ступенек:"), 0, 0)
        self.step_type_combo = QComboBox()
        self.step_type_combo.addItems([
            "Чередующиеся (±амплитуда)",
            "Случайные (±амплитуда)",
            "Синусоидальные зоны", 
            "Пилообразные зоны"
        ])
        step_layout.addWidget(self.step_type_combo, 0, 1)
        
        self.num_zones_input = self.create_parameter_row(
            step_layout, 1, "Количество зон:", "20"
        )
        self.step_height_input = self.create_parameter_row(
            step_layout, 2, "Амплитуда ступеньки (нм):", "5"
        )
        
        step_group.setLayout(step_layout)
        layout.addWidget(step_group)
        
        # Группа: Высокочастотная волнистость
        ripple_group = QGroupBox("Высокочастотная волнистость")
        ripple_layout = QGridLayout()
        
        self.ripple_freq_input = self.create_parameter_row(
            ripple_layout, 0, "Частота (cycles/mm):", "2.0"
        )
        self.ripple_amp_input = self.create_parameter_row(
            ripple_layout, 1, "Амплитуда (нм):", "3"
        )
        self.enable_ripple_check = QCheckBox("Включить волнистость")
        ripple_layout.addWidget(self.enable_ripple_check, 2, 0, 1, 2)
        
        ripple_group.setLayout(ripple_layout)
        layout.addWidget(ripple_group)
        
        # Группа: Шум
        noise_group = QGroupBox("Случайный шум")
        noise_layout = QGridLayout()
        
        self.noise_amp_input = self.create_parameter_row(
            noise_layout, 0, "Амплитуда (нм RMS):", "1.0"
        )
        self.enable_noise_check = QCheckBox("Включить шум")
        noise_layout.addWidget(self.enable_noise_check, 1, 0, 1, 2)
        
        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)
        
        # Кнопки действий
        action_layout = QVBoxLayout()
        
        generate_btn = QPushButton("Сгенерировать поверхность")
        generate_btn.clicked.connect(self.generate_surface)
        generate_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; }"
        )
        action_layout.addWidget(generate_btn)
        
        save_btn = QPushButton("Сохранить Zemax Grid Sag (.DAT)")
        save_btn.clicked.connect(self.save_zemax_format)
        action_layout.addWidget(save_btn)
        
        layout.addLayout(action_layout)
        
        # Статистика
        stats_group = QGroupBox("Статистика ошибок поверхности")
        stats_layout = QGridLayout()
        
        self.rms_label = QLabel("-")
        self.pv_label = QLabel("-")
        self.mean_label = QLabel("-")
        
        stats_layout.addWidget(QLabel("RMS (нм):"), 0, 0)
        stats_layout.addWidget(self.rms_label, 0, 1)
        stats_layout.addWidget(QLabel("PV (нм):"), 1, 0)
        stats_layout.addWidget(self.pv_label, 1, 1)
        stats_layout.addWidget(QLabel("Mean (нм):"), 2, 0)
        stats_layout.addWidget(self.mean_label, 2, 1)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel
    
    def create_parameter_row(self, layout, row, label_text, default_value):
        label = QLabel(label_text)
        input_field = QLineEdit(default_value)
        layout.addWidget(label, row, 0)
        layout.addWidget(input_field, row, 1)
        return input_field
    
    def create_plots_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Вкладки для разных видов
        tabs = QTabWidget()
        
        # 2D карта ошибок (matplotlib)
        self.canvas_2d = FigureCanvas(Figure(figsize=(8, 6)))
        self.ax_2d = self.canvas_2d.figure.add_subplot(111)
        tabs.addTab(self.canvas_2d, "2D карта ошибок")
        
        # Cross-section (matplotlib)
        self.canvas_cross = FigureCanvas(Figure(figsize=(8, 6)))
        self.ax_cross = self.canvas_cross.figure.add_subplot(111)
        tabs.addTab(self.canvas_cross, "Cross-section")

        # Grid Sag cross-section (только ошибки, увеличенный масштаб)
        self.canvas_gridsag = FigureCanvas(Figure(figsize=(8, 6)))
        self.ax_gridsag = self.canvas_gridsag.figure.add_subplot(111)
        tabs.addTab(self.canvas_gridsag, "Grid Sag Cross-section")

        # 3D ошибки (matplotlib)
        self.canvas_3d_errors = FigureCanvas(Figure(figsize=(8, 6)))
        self.ax_3d_errors = self.canvas_3d_errors.figure.add_subplot(111, projection='3d')
        tabs.addTab(self.canvas_3d_errors, "3D ошибки")
        
        # 3D полная поверхность (matplotlib)
        self.canvas_3d_full = FigureCanvas(Figure(figsize=(8, 6)))
        self.ax_3d_full = self.canvas_3d_full.figure.add_subplot(111, projection='3d')
        tabs.addTab(self.canvas_3d_full, "3D полная поверхность")
        
        layout.addWidget(tabs)
        
        return panel
    
    def compute_base_surface(self, X, Y):
        """Вычисление базовой конической поверхности"""
        radius = float(self.radius_input.text())
        conic = float(self.conic_input.text())
        
        R = np.sqrt(X**2 + Y**2)
        
        c = 1.0 / radius if radius != 0 else 0
        
        with np.errstate(invalid='ignore', divide='ignore'):
            discriminant = 1 - (1 + conic) * c**2 * R**2
            discriminant = np.maximum(discriminant, 0)
            
            Z_base = (c * R**2) / (1 + np.sqrt(discriminant))
            Z_base = np.nan_to_num(Z_base)
        
        return Z_base
    
    def generate_surface(self):
        try:
            # Чтение параметров
            aperture = float(self.aperture_input.text())
            grid_size = int(self.grid_size_input.text())
            num_zones = int(self.num_zones_input.text())
            step_height = float(self.step_height_input.text()) * 1e-6  # нм -> мм
            step_type = self.step_type_combo.currentIndex()
            
            # Создание сетки
            x = np.linspace(-aperture/2, aperture/2, grid_size)
            y = np.linspace(-aperture/2, aperture/2, grid_size)
            self.X, self.Y = np.meshgrid(x, y)
            
            # Радиус
            R = np.sqrt(self.X**2 + self.Y**2)
            
            # Маска круглой апертуры
            mask = R <= aperture/2
            
            # Вычисление базовой формы
            self.Z_base = self.compute_base_surface(self.X, self.Y)
            
            # Генерация ТОЛЬКО ошибок
            self.Z = np.zeros_like(R)
            
            # Границы зон
            zone_radii = np.linspace(0, aperture/2, num_zones + 1)
            
            if step_type == 0:  # Чередующиеся ступеньки
                for i in range(num_zones):
                    r_inner = zone_radii[i]
                    r_outer = zone_radii[i + 1]
                    zone_mask = (R >= r_inner) & (R < r_outer)
                    
                    # Чередование: четные зоны +step_height, нечетные -step_height
                    height = step_height if i % 2 == 0 else -step_height
                    self.Z[zone_mask] = height
            
            elif step_type == 1:  # Случайные ступеньки
                for i in range(num_zones):
                    r_inner = zone_radii[i]
                    r_outer = zone_radii[i + 1]
                    zone_mask = (R >= r_inner) & (R < r_outer)
                    
                    # Случайная высота от -step_height до +step_height
                    height = step_height * (2 * np.random.random() - 1)
                    self.Z[zone_mask] = height
            
            elif step_type == 2:  # Синусоидальные зоны
                k = 2 * np.pi * num_zones / aperture
                self.Z = step_height * np.sin(k * R)
            
            elif step_type == 3:  # Пилообразные зоны
                for i in range(num_zones):
                    r_inner = zone_radii[i]
                    r_outer = zone_radii[i + 1]
                    zone_mask = (R >= r_inner) & (R < r_outer)
                    
                    zone_r = R - r_inner
                    zone_width = r_outer - r_inner
                    ramp = (zone_r / zone_width) * step_height
                    self.Z[zone_mask] = ramp[zone_mask]
            
            # Добавление высокочастотной волнистости
            if self.enable_ripple_check.isChecked():
                ripple_freq = float(self.ripple_freq_input.text())
                ripple_amp = float(self.ripple_amp_input.text()) * 1e-6
                ripple = ripple_amp * np.sin(2 * np.pi * ripple_freq * R)
                self.Z += ripple
            
            # Добавление случайного шума
            if self.enable_noise_check.isChecked():
                noise_amp = float(self.noise_amp_input.text()) * 1e-6
                noise = noise_amp * np.random.randn(grid_size, grid_size)
                self.Z += noise
            
            # Применение маски
            self.Z[~mask] = 0
            
            # Обновление графиков
            self.update_plots()
            
            # Обновление статистики
            z_valid = self.Z[mask]
            rms = np.sqrt(np.mean(z_valid**2)) * 1e6
            pv = (np.max(z_valid) - np.min(z_valid)) * 1e6
            mean = np.mean(z_valid) * 1e6
            
            self.rms_label.setText(f"{rms:.3f}")
            self.pv_label.setText(f"{pv:.3f}")
            self.mean_label.setText(f"{mean:.3f}")
            
            QMessageBox.information(
                self, "Успех", 
                f"Поверхность сгенерирована!\n\n"
                f"RMS ошибки: {rms:.3f} нм\n"
                f"PV ошибки: {pv:.3f} нм"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка генерации: {str(e)}")
    
    def update_plots(self):
        if self.Z is None:
            return
        
        # 2D карта ОШИБОК (matplotlib)
        # Полностью очищаем figure и пересоздаем axes
        self.canvas_2d.figure.clear()
        self.ax_2d = self.canvas_2d.figure.add_subplot(111)

        Z_nm = self.Z * 1e6
        aperture = float(self.aperture_input.text())

        im = self.ax_2d.imshow(
            Z_nm,
            extent=[-aperture/2, aperture/2, -aperture/2, aperture/2],
            origin='lower',
            cmap='turbo',
            aspect='equal'
        )

        self.ax_2d.set_xlabel('X (мм)', fontsize=10)
        self.ax_2d.set_ylabel('Y (мм)', fontsize=10)

        # Colorbar (figure очищена, так что просто создаем новый)
        self.canvas_2d.figure.colorbar(im, ax=self.ax_2d, label='Ошибки (нм)')

        self.canvas_2d.draw()
        
        # Cross-section (matplotlib)
        self.ax_cross.clear()

        grid_size = int(self.grid_size_input.text())
        center_idx = grid_size // 2

        x_cross = self.X[center_idx, :]
        z_errors = self.Z[center_idx, :]  # в мм
        z_total = (self.Z_base[center_idx, :] + self.Z[center_idx, :])  # в мм

        r_cross = np.abs(x_cross)
        sort_idx = np.argsort(r_cross)
        r_sorted = r_cross[sort_idx]

        # График ошибок
        self.ax_cross.plot(
            r_sorted, z_errors[sort_idx],
            'r-', linewidth=2, label='Ошибки (для Grid Sag)'
        )

        # График полной поверхности
        self.ax_cross.plot(
            r_sorted, z_total[sort_idx],
            'b-', linewidth=1, label='Полная поверхность'
        )

        self.ax_cross.set_xlabel('Радиус (мм)', fontsize=10)
        self.ax_cross.set_ylabel('Sag (мм)', fontsize=10)
        self.ax_cross.legend()
        self.ax_cross.grid(True, alpha=0.3)

        # Интерактивный курсор для отображения координат
        self.canvas_cross.figure.canvas.mpl_connect('motion_notify_event', self.on_cross_hover)
        self.canvas_cross.draw()

        # Grid Sag cross-section (только ошибки, нанометровый масштаб)
        self.ax_gridsag.clear()

        # График только ошибок в нанометрах для видимости ступенек
        z_errors_nm = self.Z[center_idx, :] * 1e6  # мм -> нм

        self.ax_gridsag.plot(
            r_sorted, z_errors_nm[sort_idx],
            'r-', linewidth=2
        )

        self.ax_gridsag.set_xlabel('Радиус (мм)', fontsize=10)
        self.ax_gridsag.set_ylabel('Grid Sag (нм)', fontsize=10)
        self.ax_gridsag.grid(True, alpha=0.3)

        # Интерактивный курсор
        self.canvas_gridsag.figure.canvas.mpl_connect('motion_notify_event', self.on_gridsag_hover)
        self.canvas_gridsag.draw()

        # 3D ошибки (matplotlib)
        self.update_3d_matplotlib_errors()

        # 3D полная поверхность (matplotlib)
        self.update_3d_matplotlib_full()

    def on_cross_hover(self, event):
        """Обработчик наведения мыши на cross-section график"""
        if event.inaxes == self.ax_cross and event.xdata is not None and event.ydata is not None:
            # Обновляем заголовок с координатами
            self.ax_cross.set_title(
                f'Радиус: {event.xdata:.3f} мм, Sag: {event.ydata:.6f} мм',
                fontsize=10
            )
            self.canvas_cross.draw_idle()
        else:
            # Убираем заголовок
            self.ax_cross.set_title('')
            self.canvas_cross.draw_idle()

    def on_gridsag_hover(self, event):
        """Обработчик наведения мыши на Grid Sag cross-section график"""
        if event.inaxes == self.ax_gridsag and event.xdata is not None and event.ydata is not None:
            # Обновляем заголовок с координатами
            self.ax_gridsag.set_title(
                f'Радиус: {event.xdata:.3f} мм, Grid Sag: {event.ydata:.3f} нм',
                fontsize=10
            )
            self.canvas_gridsag.draw_idle()
        else:
            # Убираем заголовок
            self.ax_gridsag.set_title('')
            self.canvas_gridsag.draw_idle()

    def update_3d_matplotlib_errors(self):
        """3D график только ошибок"""
        # Полностью очищаем figure и пересоздаем axes
        self.canvas_3d_errors.figure.clear()
        self.ax_3d_errors = self.canvas_3d_errors.figure.add_subplot(111, projection='3d')

        grid_size = int(self.grid_size_input.text())
        
        # Прореживание для производительности
        step = max(1, grid_size // 100)
        X_sub = self.X[::step, ::step]
        Y_sub = self.Y[::step, ::step]
        Z_sub = self.Z[::step, ::step] * 1e6  # в нм
        
        # Поверхность
        surf = self.ax_3d_errors.plot_surface(
            X_sub, Y_sub, Z_sub,
            cmap='viridis',
            linewidth=0,
            antialiased=True,
            alpha=0.9,
            edgecolor='none'
        )
        
        self.ax_3d_errors.set_xlabel('X (мм)', fontsize=10)
        self.ax_3d_errors.set_ylabel('Y (мм)', fontsize=10)
        self.ax_3d_errors.set_zlabel('Ошибки (нм)', fontsize=10)
        self.ax_3d_errors.set_title('3D визуализация ошибок поверхности', fontsize=12, fontweight='bold')

        # Colorbar (figure очищена, так что просто создаем новый)
        self.canvas_3d_errors.figure.colorbar(surf, ax=self.ax_3d_errors, shrink=0.5, aspect=5, label='Ошибки (нм)')
        
        # Установка угла обзора
        self.ax_3d_errors.view_init(elev=25, azim=45)
        
        # Равные пропорции
        max_range = np.array([X_sub.max()-X_sub.min(), Y_sub.max()-Y_sub.min()]).max() / 2.0
        mid_x = (X_sub.max()+X_sub.min()) * 0.5
        mid_y = (Y_sub.max()+Y_sub.min()) * 0.5
        self.ax_3d_errors.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax_3d_errors.set_ylim(mid_y - max_range, mid_y + max_range)
        
        self.canvas_3d_errors.draw()
    
    def update_3d_matplotlib_full(self):
        """3D график полной поверхности"""
        # Полностью очищаем figure и пересоздаем axes
        self.canvas_3d_full.figure.clear()
        self.ax_3d_full = self.canvas_3d_full.figure.add_subplot(111, projection='3d')

        grid_size = int(self.grid_size_input.text())
        
        step = max(1, grid_size // 100)
        X_sub = self.X[::step, ::step]
        Y_sub = self.Y[::step, ::step]
        Z_full_sub = (self.Z_base[::step, ::step] + self.Z[::step, ::step]) * 1e3  # в мкм
        
        # Поверхность
        surf = self.ax_3d_full.plot_surface(
            X_sub, Y_sub, Z_full_sub,
            cmap='jet',
            linewidth=0,
            antialiased=True,
            alpha=0.9,
            edgecolor='none'
        )
        
        self.ax_3d_full.set_xlabel('X (мм)', fontsize=10)
        self.ax_3d_full.set_ylabel('Y (мм)', fontsize=10)
        self.ax_3d_full.set_zlabel('Высота (мкм)', fontsize=10)
        self.ax_3d_full.set_title('3D визуализация полной поверхности', fontsize=12, fontweight='bold')

        # Colorbar (figure очищена, так что просто создаем новый)
        self.canvas_3d_full.figure.colorbar(surf, ax=self.ax_3d_full, shrink=0.5, aspect=5, label='Высота (мкм)')
        
        # Установка угла обзора
        self.ax_3d_full.view_init(elev=25, azim=45)
        
        # Равные пропорции
        max_range = np.array([X_sub.max()-X_sub.min(), Y_sub.max()-Y_sub.min()]).max() / 2.0
        mid_x = (X_sub.max()+X_sub.min()) * 0.5
        mid_y = (Y_sub.max()+Y_sub.min()) * 0.5
        self.ax_3d_full.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax_3d_full.set_ylim(mid_y - max_range, mid_y + max_range)
        
        self.canvas_3d_full.draw()
    
    def save_zemax_format(self):
        if self.Z is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала сгенерируйте поверхность!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить Zemax Grid Sag", "", "DAT Files (*.dat)"
        )
        
        if filename:
            try:
                grid_size = int(self.grid_size_input.text())
                aperture = float(self.aperture_input.text())
                
                delx = aperture / (grid_size - 1)
                dely = delx
                
                with open(filename, 'w') as f:
                    # Заголовок
                    f.write(f"{grid_size} {grid_size} {delx:.6f} {dely:.6f} 0 0.0 0.0\n")
                    
                    # Данные: слева направо, сверху вниз
                    for i in range(grid_size):
                        for j in range(grid_size):
                            z_val = self.Z[i, j]
                            f.write(f"{z_val:.8e} 0.0 0.0 0.0 0\n")

                QMessageBox.information(
                    self, "Успех",
                    f"Файл сохранен:\n{filename}"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MSFEGenerator()
    window.show()
    
    sys.exit(app.exec())