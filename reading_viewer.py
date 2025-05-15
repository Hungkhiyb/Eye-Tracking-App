from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QPalette, QFont, QPainter, QPen
import time

class ReadingViewerWidget(QWidget):
    back_to_menu = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Thiết lập nền tối
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#1A1A1A"))
        self.setPalette(palette)

        # Cấu hình gaze tracking
        self.gaze_point = QPoint(-100, -100)
        self.scroll_speed = 15
        self.dwell_threshold = 1.2
        self.zone_activated = False
        self.dwell_start_time = 0
        self.left_zone_width = 100  # Độ rộng vùng bên trái để back

        # Khởi tạo layout và scroll area
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: #2C3E50;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3498DB;
                min-height: 20px;
                border-radius: 6px;
            }
        """)
        
        # Nội dung cuộn
        self.content_label = QLabel()
        self.content_label.setWordWrap(True)
        self.content_label.setAlignment(Qt.AlignTop)
        self.content_label.setStyleSheet("""
            color: #ECF0F1;
            font: 18pt Arial;
            line-height: 1.6;
            padding: 20px 300px;
            background: transparent;
        """)
        self.scroll_area.setWidget(self.content_label)
        self.main_layout.addWidget(self.scroll_area)

        # Thiết lập timer
        self.scroll_timer = QTimer()
        self.scroll_timer.timeout.connect(self.handle_auto_scroll)
        self.scroll_timer.start(50)

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def load_article(self, text):
        lines = text.split('\n')
        title = lines[0] if lines else ''
        body = '<br><br>'.join(lines[1:])

        html_content = f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: Arial;
                        color: #ECF0F1;
                        padding: 30px 80px;
                        font-size: 18pt;
                        line-height: 1.8;
                        text-align: justify;
                    }}
                    .title {{
                        text-align: center;
                        font-size: 26pt;
                        font-weight: bold;
                        margin-bottom: 40px;
                    }}
                </style>
            </head>
            <body>
                <div class='title'>{title}</div>
                <div>{body}</div>
            </body>
        </html>
        """

        self.content_label.setText(html_content)
        self.scroll_area.verticalScrollBar().setValue(0)

    def update_gaze(self, x, y):
        # Cập nhật tọa độ gaze (từ hệ thống tracking)
        self.gaze_point = self.mapFromGlobal(QPoint(x, y))

    def handle_auto_scroll(self):
        if self.gaze_point.x() < 0: 
            return

        # Tự động cuộn
        scroll_bar = self.scroll_area.verticalScrollBar()
        y_pos = self.gaze_point.y()
        
        if y_pos < self.height() // 5:
            scroll_bar.setValue(scroll_bar.value() - self.scroll_speed)
        elif y_pos > 4 * self.height() // 5:
            scroll_bar.setValue(scroll_bar.value() + self.scroll_speed)

        # Kiểm tra nếu nhìn sang bên trái màn hình
        if self.gaze_point.x() < self.left_zone_width:
            if not self.zone_activated:
                if time.time() - self.dwell_start_time > self.dwell_threshold:
                    self.back_to_menu.emit()
                    self.zone_activated = True
        else:
            self.dwell_start_time = time.time()
            self.zone_activated = False

    def paintEvent(self, event):
        super().paintEvent(event)
        if hasattr(self, 'gaze_point'):
            # Vẽ điểm gaze
            painter = QPainter(self)
            
            # Inner red circle
            painter.setBrush(QColor(255, 0, 0, 180))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.gaze_point.x() - 8, self.gaze_point.y() - 8, 16, 16)

            # Outer white ring
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self.gaze_point.x() - 10, self.gaze_point.y() - 10, 20, 20)

            # Vẽ vùng bên trái (back zone) nếu cần debug
            debug_mode = False  # Đặt thành True để xem vùng back
            if debug_mode:
                painter.setPen(QPen(Qt.green, 2))
                painter.setBrush(QColor(0, 255, 0, 50))
                painter.drawRect(0, 0, self.left_zone_width, self.height())