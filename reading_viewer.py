from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QScrollArea
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPainterPath, QPalette, QLinearGradient
import time

class ReadingViewerWidget(QWidget):
    back_to_menu = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#1A1A1A"))
        self.setPalette(palette)

        # Scroll control
        self.gaze_x = -100
        self.gaze_y = -100
        self.scroll_speed = 15  # Tăng tốc độ cuộn
        self.dwell_threshold = 1.2  # Giảm thời gian dwell

        # Nút back
        self.back_button_size = 100  # Tăng kích thước
        self.back_button_rect = QRect(0, 0, 0, 0)
        self.corner_radius = 20

        # === Layout ===
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area với thanh cuộn rõ ràng
        self.scroll_area = QScrollArea()
        self.scroll_content = QLabel()

        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #1A1A1A;  /* Màu nền tối */
                border: none;
            }
            QScrollBar:vertical {
                background: #2C3E50;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #3498DB;
                min-height: 20px;
                border-radius: 6px;
            }
        """)

        self.scroll_content.setStyleSheet("""
            color: #ECF0F1;  /* Màu chữ trắng */
            font-size: 18px;
            line-height: 1.6;
            padding: 30px;
            background-color: transparent;  /* Nền trong suốt */
            font-family: Arial;  /* Font chữ rõ ràng */
        """)

        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content.setWordWrap(True)
        layout.addWidget(self.scroll_area)

        font = QFont("Arial", 18)  # Font Arial, cỡ 18px
        self.scroll_content.setFont(font)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_scroll)
        self.timer.start(50)

    def resizeEvent(self, event):
        """Cập nhật vị trí nút back"""
        self.back_button_rect = QRect(
            self.width() - self.back_button_size,  # X: sát lề phải
            0,  # Y: sát lề trên
            self.back_button_size,
            self.back_button_size
        )
        super().resizeEvent(event)

    def load_article(self, text):
        self.scroll_content.setText(text)  # Đặt nội dung văn bản
        self.scroll_content.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # Căn lề trái
        self.scroll_area.verticalScrollBar().setValue(0)  # Reset thanh cuộn

    def update_gaze(self, x, y):
        self.gaze_x = x
        self.gaze_y = y

    def update_scroll(self):
        # Bỏ qua nếu gaze chưa hợp lệ
        if self.gaze_x < 0 or self.gaze_y < 0:
            return

        # Xử lý scroll
        h = self.height()
        top_zone = h // 3
        bottom_zone = 2 * h // 3

        if self.gaze_y < top_zone:
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - self.scroll_speed
            )
            self.current_zone = 'up'
        elif self.gaze_y > bottom_zone:
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() + self.scroll_speed
            )
            self.current_zone = 'down'
        else:
            self.current_zone = 'hold'

        # Kiểm tra nút back
        if self.back_button_rect.contains(self.gaze_x, self.gaze_y):
            if not self.zone_activated:
                if time.time() - self.dwell_start_time > self.dwell_threshold:
                    self.zone_activated = True
                    self.back_to_menu.emit()
                    self.dwell_start_time = time.time()
        else:
            self.dwell_start_time = time.time()
            self.zone_activated = False

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Vẽ nền widget
        painter.fillRect(self.rect(), QColor("#1A1A1A"))  # Màu nền tối
        
        # Vẽ nút back (hình vuông bo tròn)
        path = QPainterPath()
        rect_f = QRectF(self.back_button_rect)
        path.addRoundedRect(rect_f, self.corner_radius, self.corner_radius)
        
        gradient = QLinearGradient(rect_f.topLeft(), rect_f.bottomRight())
        gradient.setColorAt(0, QColor("#E74C3C"))
        gradient.setColorAt(1, QColor("#C0392B"))
        painter.setBrush(gradient)
        painter.setPen(QPen(Qt.white, 3))
        painter.drawPath(path)

        # Mũi tên và text
        painter.setPen(Qt.white)
        font = QFont("Arial", 24, QFont.Bold)
        painter.setFont(font)
        painter.drawText(self.back_button_rect, Qt.AlignCenter, "⬅")

        # Điểm nhìn
        painter.setBrush(QColor(255, 255, 255, 150))
        painter.drawEllipse(int(self.gaze_x) - 10, int(self.gaze_y) - 10, 20, 20)

    def sizeHint(self):
        return self.parent().size() if self.parent() else super().sizeHint()