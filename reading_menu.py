from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtCore import Qt, QRect, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QPalette
import time

class ReadingMenuWidget(QWidget):
    back_to_home = pyqtSignal()
    article_selected = pyqtSignal(int)
    turn_page = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self.dwell_threshold = 1.5
        self.gaze_x = -100
        self.gaze_y = -100
        self.current_zone = None
        self.dwell_start_time = None
        self.zone_activated = False
        self.zones = []  # Thêm dòng này để khởi tạo danh sách zones

        # Thiết lập màu nền
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#1A1A1A"))
        self.setPalette(palette)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def update_gaze(self, x, y):
        self.gaze_x = x
        self.gaze_y = y

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        self.zones = []  # Reset danh sách zones mỗi lần vẽ

        # === TOP BAR (Quay về) ===
        top_rect = QRect(0, 0, w, 80)
        self.zones.append(("top", top_rect))  # Thêm vào danh sách zones
        gradient = QLinearGradient(top_rect.topLeft(), top_rect.bottomLeft())
        gradient.setColorAt(0, QColor("#2C3E50"))
        gradient.setColorAt(1, QColor("#34495E"))
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawRect(top_rect)

        # Vẽ chữ "Quay về"
        painter.setPen(QColor("#ECF0F1"))
        font = QFont("Arial", 20, QFont.Bold)
        painter.setFont(font)
        painter.drawText(top_rect, Qt.AlignCenter, "⬅ Quay về trang chủ")

        # === LƯỚI BÀI BÁO (2x2) ===
        grid_margin = 30
        cell_w = (w - 3 * grid_margin) // 2
        cell_h = (h - 120 - grid_margin) // 2

        for row in range(2):
            for col in range(2):
                idx = row * 2 + col
                x = grid_margin + col * (cell_w + grid_margin)
                y = 100 + row * (cell_h + grid_margin)
                rect = QRect(x, y, cell_w, cell_h)
                self.zones.append((f"article_{idx}", rect))  # Thêm vào zones

                # Vẽ ô bài báo
                painter.setBrush(QColor("#3498DB" if rect.contains(self.gaze_x, self.gaze_y) else "#2980B9"))
                painter.setPen(QPen(QColor("#2C3E50"), 3))
                painter.drawRoundedRect(rect, 15, 15)

                # Vẽ tiêu đề
                painter.setPen(Qt.white)
                font.setPointSize(16)
                painter.drawText(rect.adjusted(10, 10, -10, -10), Qt.AlignCenter, f"BÀI BÁO {idx + 1}")

        # === NÚT CHUYỂN TRANG ===
        arrow_size = 80
        left_rect = QRect(20, h // 2 - arrow_size // 2, arrow_size, arrow_size)
        right_rect = QRect(w - 20 - arrow_size, h // 2 - arrow_size // 2, arrow_size, arrow_size)
        self.zones.extend([("left", left_rect), ("right", right_rect)])  # Thêm vào zones

        # Vẽ nút
        for rect, text in [(left_rect, "←"), (right_rect, "→")]:
            painter.setBrush(QColor("#E74C3C" if rect.contains(self.gaze_x, self.gaze_y) else "#C0392B"))
            painter.drawRoundedRect(rect, 15, 15)
            painter.setPen(Qt.white)
            painter.drawText(rect, Qt.AlignCenter, text)

        # === ĐIỂM NHÌN ===
        painter.setBrush(QColor(255, 255, 255, 150))
        painter.drawEllipse(int(self.gaze_x) - 10, int(self.gaze_y) - 10, 20, 20)

        self.check_gaze_zone()

    def check_gaze_zone(self):
        hit_zone = None
        for name, rect in self.zones:  # Truy cập zones đã được khởi tạo
            if rect.contains(self.gaze_x, self.gaze_y):
                hit_zone = name
                break

        if hit_zone != self.current_zone:
            self.current_zone = hit_zone
            self.dwell_start_time = time.time()
            self.zone_activated = False
        else:
            if hit_zone and not self.zone_activated:
                if time.time() - self.dwell_start_time >= self.dwell_threshold:
                    self.zone_activated = True
                    self.activate_zone(hit_zone)

    def activate_zone(self, zone_name):
        if zone_name == "top":
            self.back_to_home.emit()
        elif zone_name == "left":
            self.turn_page.emit("left")
        elif zone_name == "right":
            self.turn_page.emit("right")
        elif zone_name.startswith("article_"):
            idx = int(zone_name.split("_")[1])
            self.article_selected.emit(idx)

    def sizeHint(self):
        return self.parent().size() if self.parent() else super().sizeHint()