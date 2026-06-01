#!/usr/bin/env python3

import sys
import math
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QPointF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPainterPath
)

C_BG         = QColor(10,  14,  26)
C_GRID       = QColor(25,  45,  80)
C_TEXT       = QColor(200, 210, 230)
C_ACCENT     = QColor(60,  140, 255)
C_SAG_IDLE   = QColor(20,  80,  180, 60)
C_ORTA_IDLE  = QColor(20,  150, 100, 60)
C_SOL_IDLE   = QColor(20,  80,  180, 60)
C_SAG_ENGEL  = QColor(255, 80,  60,  130)
C_ORTA_ENGEL = QColor(255, 160, 20,  130)
C_SOL_ENGEL  = QColor(255, 80,  60,  130)
C_LIDAR_PT   = QColor(0,   220, 255)
C_ROBOT      = QColor(100, 180, 255)


class RosSignals(QObject):
    scan_received   = pyqtSignal(object)
    sag_engel_recv  = pyqtSignal(float)
    orta_engel_recv = pyqtSignal(float)
    sol_engel_recv  = pyqtSignal(float)
    sag_mes_recv    = pyqtSignal(float)
    orta_mes_recv   = pyqtSignal(float)
    sol_mes_recv    = pyqtSignal(float)


class LidarGuiSubscriber(Node):
    def __init__(self, signals: RosSignals):
        super().__init__('lidar_gui_node')
        self.signals = signals
        qos = rclpy.qos.qos_profile_sensor_data

        self.create_subscription(LaserScan, '/lidar_gui_raw', self._on_scan, qos_profile=qos)
        self.create_subscription(Float32, '/sag_engel',  lambda m: signals.sag_engel_recv.emit(m.data),  10)
        self.create_subscription(Float32, '/orta_engel', lambda m: signals.orta_engel_recv.emit(m.data), 10)
        self.create_subscription(Float32, '/sol_engel',  lambda m: signals.sol_engel_recv.emit(m.data),  10)
        self.create_subscription(Float32, '/sag_mesafe', lambda m: signals.sag_mes_recv.emit(m.data),    10)
        self.create_subscription(Float32, '/orta_mesafe',lambda m: signals.orta_mes_recv.emit(m.data),   10)
        self.create_subscription(Float32, '/sol_mesafe', lambda m: signals.sol_mes_recv.emit(m.data),    10)

    def _on_scan(self, msg):
        self.signals.scan_received.emit(msg)


class LidarFanWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sag_engel   = 0.0
        self.orta_engel  = 0.0
        self.sol_engel   = 0.0
        self.sag_mesafe  = 0.0
        self.orta_mesafe = 0.0
        self.sol_mesafe  = 0.0
        self.scan_points: list[tuple[float, float]] = []
        self.max_range_m = 5.0

    def update_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        valid  = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        r, a   = ranges[valid], angles[valid]
        x, y   = r * np.cos(a), r * np.sin(a)
        front  = y >= 0
        self.scan_points = list(zip(x[front].tolist(), y[front].tolist()))
        self.update()

    def update_zones(self, sag_e, orta_e, sol_e, sag_m, orta_m, sol_m):
        self.sag_engel   = sag_e
        self.orta_engel  = orta_e
        self.sol_engel   = sol_e
        self.sag_mesafe  = sag_m
        self.orta_mesafe = orta_m
        self.sol_mesafe  = sol_m
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H   = self.width(), self.height()
        cx, cy = W // 2, H - 40
        radius = min(W // 2 - 30, H - 60)
        p.fillRect(0, 0, W, H, C_BG)
        self._draw_grid(p, cx, cy, radius)
        self._draw_zones(p, cx, cy, radius)
        self._draw_points(p, cx, cy, radius)
        self._draw_robot(p, cx, cy)
        self._draw_labels(p, cx, cy, radius)
        p.end()

    def _draw_grid(self, p, cx, cy, radius):
        p.setPen(QPen(C_GRID, 1, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        steps = int(self.max_range_m)
        for i in range(1, steps + 1):
            r_px = int(radius * i / self.max_range_m)
            p.drawArc(cx - r_px, cy - r_px, r_px * 2, r_px * 2, 0 * 16, 180 * 16)
        p.setPen(QPen(C_GRID, 1, Qt.SolidLine))
        for deg in range(0, 181, 30):
            rad = math.radians(deg)
            p.drawLine(cx, cy, int(cx + radius * math.cos(math.pi - rad)),
                       int(cy - radius * math.sin(math.pi - rad)))
        p.setFont(QFont('Segoe UI', 7))
        p.setPen(QPen(C_GRID.lighter(160)))
        for i in range(1, steps + 1):
            r_px = int(radius * i / self.max_range_m)
            p.drawText(cx + 3, cy - r_px + 10, f'{i}m')

    def _draw_zones(self, p, cx, cy, radius):
        def draw_sector(start_deg, end_deg, color):
            path = QPainterPath()
            path.moveTo(cx, cy)
            path.arcTo(cx - radius, cy - radius, radius * 2, radius * 2,
                       180 - end_deg, end_deg - start_deg)
            path.closeSubpath()
            p.fillPath(path, QBrush(color))
            p.setPen(QPen(color.lighter(150), 1))
            p.drawPath(path)

        draw_sector(0,   60,  C_SAG_ENGEL  if self.sag_engel  else C_SAG_IDLE)
        draw_sector(60,  120, C_ORTA_ENGEL if self.orta_engel else C_ORTA_IDLE)
        draw_sector(120, 180, C_SOL_ENGEL  if self.sol_engel  else C_SOL_IDLE)

    def _draw_points(self, p, cx, cy, radius):
        if not self.scan_points:
            return
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(C_LIDAR_PT))
        scale = radius / self.max_range_m
        for mx, my in self.scan_points:
            dist = math.hypot(mx, my)
            if dist > self.max_range_m or dist < 0.01:
                continue
            angle_deg = math.degrees(math.atan2(my, mx))
            if angle_deg < 0 or angle_deg > 180:
                continue
            px = cx + dist * math.cos(math.pi - math.radians(angle_deg)) * scale
            py = cy - dist * math.sin(math.pi - math.radians(angle_deg)) * scale
            p.drawEllipse(QPointF(px, py), 2.2, 2.2)

    def _draw_robot(self, p, cx, cy):
        p.setPen(QPen(C_ROBOT, 2))
        p.setBrush(QBrush(QColor(30, 60, 110)))
        p.drawEllipse(cx - 10, cy - 10, 20, 20)
        p.setBrush(QBrush(C_ACCENT))
        p.setPen(Qt.NoPen)
        p.drawEllipse(cx - 4, cy - 4, 8, 8)
        p.setPen(QPen(C_ROBOT, 2))
        p.drawLine(cx, cy - 10, cx, cy - 24)
        p.drawLine(cx, cy - 24, cx - 5, cy - 17)
        p.drawLine(cx, cy - 24, cx + 5, cy - 17)

    def _draw_labels(self, p, cx, cy, radius):
        font_big   = QFont('Segoe UI', 11, QFont.Bold)
        font_small = QFont('Segoe UI', 8)

        def zone_label(center_deg, name, engel, mesafe, color):
            rad = math.radians(center_deg)
            lx  = cx + (radius * 0.60) * math.cos(math.pi - rad)
            ly  = cy - (radius * 0.60) * math.sin(math.pi - rad)
            p.setFont(font_big)
            p.setPen(QPen(color))
            p.drawText(int(lx) - 28, int(ly) - 6,  56, 18, Qt.AlignCenter, name)
            p.setFont(font_small)
            p.setPen(QPen(C_TEXT))
            p.drawText(int(lx) - 28, int(ly) + 10, 56, 14, Qt.AlignCenter,
                       'ENGEL' if engel else 'TEMİZ')
            p.drawText(int(lx) - 28, int(ly) + 22, 56, 14, Qt.AlignCenter,
                       f'{mesafe:.2f}m' if engel else '—')

        zone_label(30,  'SAĞ',  self.sag_engel,  self.sag_mesafe,
                   C_SAG_ENGEL.lighter(180)  if self.sag_engel  else C_TEXT)
        zone_label(90,  'ORTA', self.orta_engel, self.orta_mesafe,
                   C_ORTA_ENGEL.lighter(180) if self.orta_engel else C_TEXT)
        zone_label(150, 'SOL',  self.sol_engel,  self.sol_mesafe,
                   C_SOL_ENGEL.lighter(180)  if self.sol_engel  else C_TEXT)

        p.setFont(QFont('Segoe UI', 7))
        p.setPen(QPen(C_GRID.lighter(200)))
        for deg, label in [(0, '0°'), (60, '60°'), (120, '120°'), (180, '180°')]:
            rad = math.radians(deg)
            ex  = cx + (radius + 14) * math.cos(math.pi - rad)
            ey  = cy - (radius + 14) * math.sin(math.pi - rad)
            p.drawText(int(ex) - 15, int(ey) - 7, 30, 14, Qt.AlignCenter, label)


class ZoneCard(QFrame):
    def __init__(self, title, color_hex, parent=None):
        super().__init__(parent)
        self.setFixedHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.base_color = color_hex
        self.setStyleSheet(f'QFrame {{ background: #0e1628; border: 1.5px solid {color_hex}55; border-radius: 12px; }}')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setFont(QFont('Segoe UI', 12, QFont.Bold))
        title_lbl.setStyleSheet(f'color: {color_hex}; background: transparent; border: none;')
        layout.addWidget(title_lbl)

        self.status_lbl = QLabel('TEMİZ')
        self.status_lbl.setFont(QFont('Segoe UI', 18, QFont.Bold))
        self.status_lbl.setStyleSheet('color: #4cde8a; background: transparent; border: none;')
        layout.addWidget(self.status_lbl)

        self.dist_lbl = QLabel('—')
        self.dist_lbl.setFont(QFont('Segoe UI', 11))
        self.dist_lbl.setStyleSheet('color: #8899bb; background: transparent; border: none;')
        layout.addWidget(self.dist_lbl)

    def update_data(self, engel: float, mesafe: float):
        if engel > 0.5:
            self.status_lbl.setText('ENGEL VAR ⚠')
            self.status_lbl.setStyleSheet('color: #ff5040; background: transparent; border: none;')
            self.dist_lbl.setText(f'{mesafe:.2f} m')
            self.setStyleSheet(f'QFrame {{ background: #1a0a0a; border: 1.5px solid {self.base_color}; border-radius: 12px; }}')
        else:
            self.status_lbl.setText('TEMİZ ✓')
            self.status_lbl.setStyleSheet('color: #4cde8a; background: transparent; border: none;')
            self.dist_lbl.setText('—')
            self.setStyleSheet(f'QFrame {{ background: #0e1628; border: 1.5px solid {self.base_color}55; border-radius: 12px; }}')


class MainWindow(QMainWindow):
    def __init__(self, signals: RosSignals):
        super().__init__()
        self.setWindowTitle('LiDAR Engel Görselleştirici')
        self.setMinimumSize(860, 620)
        self.setStyleSheet(f'QMainWindow {{ background: {C_BG.name()}; }}')

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel('🔍  LiDAR Engel Haritası')
        title.setFont(QFont('Segoe UI', 16, QFont.Bold))
        title.setStyleSheet(f'color: {C_ACCENT.name()}; background: transparent;')
        root.addWidget(title)

        subtitle = QLabel('0°–60° → SAĞ  |  60°–120° → ORTA  |  120°–180° → SOL')
        subtitle.setFont(QFont('Segoe UI', 9))
        subtitle.setStyleSheet('color: #5577aa; background: transparent;')
        root.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color: #1e3260;')
        root.addWidget(sep)

        self.fan = LidarFanWidget()
        root.addWidget(self.fan, stretch=1)

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)
        self.card_sag  = ZoneCard('SAĞ  (0° – 60°)',    '#3c8cff')
        self.card_orta = ZoneCard('ORTA  (60° – 120°)', '#ffa020')
        self.card_sol  = ZoneCard('SOL  (120° – 180°)', '#3c8cff')
        cards_layout.addWidget(self.card_sag)
        cards_layout.addWidget(self.card_orta)
        cards_layout.addWidget(self.card_sol)
        root.addLayout(cards_layout)

        self.status_bar = QLabel('⏳  lidar_node bekleniyor...')
        self.status_bar.setFont(QFont('Segoe UI', 8))
        self.status_bar.setStyleSheet('color: #556688; background: transparent; padding: 2px 0;')
        root.addWidget(self.status_bar)

        signals.scan_received.connect(self._on_scan)
        signals.sag_engel_recv.connect(self._on_sag_engel)
        signals.orta_engel_recv.connect(self._on_orta_engel)
        signals.sol_engel_recv.connect(self._on_sol_engel)
        signals.sag_mes_recv.connect(self._on_sag_mes)
        signals.orta_mes_recv.connect(self._on_orta_mes)
        signals.sol_mes_recv.connect(self._on_sol_mes)

        self._sag_e = self._orta_e = self._sol_e = 0.0
        self._sag_m = self._orta_m = self._sol_m = 0.0
        self._scan_count = 0

    def _on_scan(self, msg):
        self.fan.update_scan(msg)
        self._scan_count += 1
        self.status_bar.setText(
            f"✅  Bağlı  |  Tarama #{self._scan_count}  |  "
            f"SAĞ: {'ENGEL' if self._sag_e else 'TEMİZ'}  "
            f"ORTA: {'ENGEL' if self._orta_e else 'TEMİZ'}  "
            f"SOL: {'ENGEL' if self._sol_e else 'TEMİZ'}"
        )

    def _on_sag_engel(self, v):   self._sag_e  = v; self._update_all()
    def _on_orta_engel(self, v):  self._orta_e = v; self._update_all()
    def _on_sol_engel(self, v):   self._sol_e  = v; self._update_all()
    def _on_sag_mes(self, v):     self._sag_m  = v; self._update_all()
    def _on_orta_mes(self, v):    self._orta_m = v; self._update_all()
    def _on_sol_mes(self, v):     self._sol_m  = v; self._update_all()

    def _update_all(self):
        self.fan.update_zones(self._sag_e, self._orta_e, self._sol_e,
                              self._sag_m, self._orta_m, self._sol_m)
        self.card_sag.update_data(self._sag_e,  self._sag_m)
        self.card_orta.update_data(self._orta_e, self._orta_m)
        self.card_sol.update_data(self._sol_e,  self._sol_m)


def main():
    rclpy.init()
    signals   = RosSignals()
    ros_node  = LidarGuiSubscriber(signals)
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow(signals)
    window.show()
    exit_code = app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
