from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtCore import Qt, QRect, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QPalette, QPixmap, QBrush, QPalette
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

        self.all_articles = [f"BÀI BÁO {i+1}" for i in range(10)]  # Danh sách 10 bài
        self.articles_per_page = 4
        self.current_page = 0


    def update_gaze(self, x, y):
        self.gaze_x = x
        self.gaze_y = y

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        self.zones = []

        # === Load ảnh ===
        icon_back = QPixmap("assets/home_icon.png")
        icon_left = QPixmap("assets/left_arrow.png")
        icon_right = QPixmap("assets/right_arrow.png")
        book_icon = QPixmap("assets/book_icon.png")

        # === TOP BAR (Back) ===
        top_rect = QRect(0, 0, w, 150)
        self.zones.append(("top", top_rect))
        painter.setBrush(QColor(66, 165, 245, 255))
        painter.setPen(Qt.NoPen)
        painter.drawRect(top_rect)

        hover_top = top_rect.contains(self.gaze_x, self.gaze_y)
        if hover_top:
            painter.setBrush(QColor(0, 0, 255, 255))  # xám nhạt
            painter.setPen(Qt.NoPen)
            painter.drawRect(top_rect)

        icon_size = 150
        icon_rect = QRect(
            top_rect.center().x() - icon_size // 2,
            top_rect.center().y() - icon_size // 2,
            icon_size, icon_size
        )
        painter.drawPixmap(icon_rect, icon_back.scaled(icon_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # === GRID BÀI BÁO (2x2) ===
        grid_margin = 150
        cell_w = (w - 3 * grid_margin) // 2
        cell_h = (h - 120 - grid_margin) // 2

        # === GRID BÀI BÁO THEO TRANG ===
        start_idx = self.current_page * self.articles_per_page
        end_idx = start_idx + self.articles_per_page
        visible_articles = self.all_articles[start_idx:end_idx]

        for i, title in enumerate(visible_articles):
            row, col = divmod(i, 2)
            x = grid_margin + col * (cell_w + grid_margin)
            y = 150 + row * (cell_h + grid_margin)
            rect = QRect(x, y, cell_w, cell_h)
            self.zones.append((f"article_{start_idx + i}", rect))

            hover = rect.contains(self.gaze_x, self.gaze_y)
            if hover:
                painter.setBrush(QColor(255, 255, 255, 40))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(rect, 20, 20)

            icon_size = min(rect.width(), rect.height()) // 2
            icon_rect = QRect(rect.center().x() - icon_size // 2, rect.top() + 20, icon_size, icon_size)
            painter.drawPixmap(icon_rect, book_icon.scaled(icon_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            text_rect = QRect(rect.left(), icon_rect.bottom() + 10, rect.width(), 40)
            painter.setPen(Qt.white)
            font = QFont("Arial", 14, QFont.Bold)
            painter.setFont(font)
            painter.drawText(text_rect, Qt.AlignCenter, title)

        # === NÚT TRÁI / PHẢI ===
        arrow_size = 150
        left_rect = QRect(0, 150, arrow_size, h - 150)
        right_rect = QRect(w - arrow_size, 150, arrow_size, h - 150)
        self.zones.extend([("left", left_rect), ("right", right_rect)])

        for rect, icon, name in [(left_rect, icon_left, "left"), (right_rect, icon_right, "right")]:
            hover = rect.contains(self.gaze_x, self.gaze_y)
            if hover:
                painter.setBrush(QColor(255, 255, 255, 40))  # xám nhạt
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(rect, 15, 15)

            icon_rect = QRect(
                rect.center().x() - 30,
                rect.center().y() - 30,
                60, 60
            )
            painter.drawPixmap(icon_rect, icon.scaled(icon_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # === GAZE POINT + VÒNG CHỜ ===
        if hasattr(self, 'gaze_x') and hasattr(self, 'gaze_y'):
            painter.setBrush(QColor(255, 0, 0, 180))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.gaze_x - 8, self.gaze_y - 8, 16, 16)

            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self.gaze_x - 10, self.gaze_y - 10, 20, 20)

            if self.current_zone is not None and not self.zone_activated:
                dwell_time = time.time() - self.dwell_start_time
                if dwell_time < self.dwell_threshold:
                    progress = dwell_time / self.dwell_threshold
                    painter.setPen(QPen(Qt.cyan, 3))
                    painter.drawArc(
                        self.gaze_x - 15, self.gaze_y - 15, 30, 30,
                        0, int(progress * 360 * 16)
                    )

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
            total_pages = (len(self.all_articles) + self.articles_per_page - 1) // self.articles_per_page
            self.current_page = (self.current_page - 1) % total_pages
            self.update()

        elif zone_name == "right":
            total_pages = (len(self.all_articles) + self.articles_per_page - 1) // self.articles_per_page
            self.current_page = (self.current_page + 1) % total_pages
            self.update()
        elif zone_name.startswith("article_"):
            idx = int(zone_name.split("_")[1])
            self.article_selected.emit(idx)

    def sizeHint(self):
        return self.parent().size() if self.parent() else super().sizeHint()