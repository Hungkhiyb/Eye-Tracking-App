# reading_viewer.py
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QScrollArea, QFrame, QApplication
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect, QRectF, QPoint, QSize
from PyQt5.QtGui import (QPainter, QColor, QPen, QFont, QPainterPath,
                         QPalette, QLinearGradient, QFontMetrics)

import time

class ReadingViewerWidget(QWidget):
    back_to_menu = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#181A1B"))
        self.setPalette(palette)

        self.gaze_x = -100
        self.gaze_y = -100
        self.scroll_speed = 20
        self.dwell_threshold_button = 1.3
        self.dwell_threshold_scroll = 0.25
        self.current_hovered_control = None
        self.dwell_start_time = None
        self.control_activated_button = False

        self.back_button_rect = QRect() # Sẽ được tính toán
        self.scroll_up_rect = QRect()
        self.scroll_down_rect = QRect()
        self.control_button_font = QFont("Arial", 14, QFont.Bold)

        self.main_v_layout = QVBoxLayout(self)
        self.main_v_layout.setContentsMargins(0, 0, 0, 0)
        self.main_v_layout.setSpacing(0)

        self.control_bar_frame = QFrame(self)
        self.control_bar_frame.setFixedHeight(60)
        self.control_bar_frame.setStyleSheet("""
            QFrame {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                 stop:0 #2A2C2B, stop:1 #222423);
                border-bottom: 1px solid #101010;
            }
        """)
        # Không cần layout cho control_bar_frame vì nút sẽ được vẽ thủ công trong paintEvent

        self.scroll_area = QScrollArea()
        self.scroll_content_label = QLabel()

        self.scroll_area.setStyleSheet("""
            QScrollArea { background-color: transparent; border: none; }
            QScrollBar:vertical { background: #282C2E; width: 16px; margin: 0px 2px 0px 0px; border-radius: 8px; }
            QScrollBar::handle:vertical { background: #5A7E9E; min-height: 40px; border-radius: 8px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; border: none; background: none; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)
        self.scroll_content_label.setStyleSheet("""
            QLabel {
                color: #D0D0D0; font-size: 21px; line-height: 1.65;
                padding: 45px; padding-top: 25px; background-color: transparent;
                font-family: "Georgia", "Times New Roman", serif;
            }
        """)

        self.scroll_area.setWidget(self.scroll_content_label)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content_label.setWordWrap(True)
        self.scroll_content_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.main_v_layout.addWidget(self.control_bar_frame)
        self.main_v_layout.addWidget(self.scroll_area, 1)

        self.gaze_interaction_timer = QTimer(self)
        self.gaze_interaction_timer.timeout.connect(self.process_gaze_interactions_viewer)
        self.gaze_interaction_timer.start(50)

        self._last_viewer_width = 0
        self._last_viewer_height = 0
        self.is_layout_calculated = False # Cờ để đảm bảo layout được tính ít nhất một lần


    def calculate_viewer_layout_rects(self):
        width = self.width()
        if width <=0 : return # Chưa có kích thước hợp lệ

        # Nút Quay lại Menu (trong control_bar_frame)
        bar_h = self.control_bar_frame.height() # Luôn là 60
        fm = QFontMetrics(self.control_button_font)
        back_text = "❮ Menu"
        text_w = fm.boundingRect(back_text).width() + 45 # Tăng padding cho dễ click
        button_h = bar_h - 20 # Margin 10px trên và dưới trong thanh bar
        
        # Đặt nút ở giữa theo chiều dọc của control_bar_frame và cách lề trái
        button_y = (bar_h - button_h) // 2
        self.back_button_rect = QRect(15, button_y, text_w, button_h)
        # print(f"DEBUG: Calculated Back Button Rect: {self.back_button_rect}")


        # Vùng cuộn
        scroll_area_geom = self.scroll_area.geometry()
        if scroll_area_geom.isValid() and scroll_area_geom.height() > 50:
            scroll_zone_h_ratio = 0.22
            scroll_trigger_h = int(scroll_area_geom.height() * scroll_zone_h_ratio)
            if scroll_trigger_h > 15:
                self.scroll_up_rect = QRect(scroll_area_geom.x(), scroll_area_geom.y(),
                                            scroll_area_geom.width(), scroll_trigger_h)
                self.scroll_down_rect = QRect(scroll_area_geom.x(),
                                              scroll_area_geom.y() + scroll_area_geom.height() - scroll_trigger_h,
                                              scroll_area_geom.width(), scroll_trigger_h)
            else:
                self.scroll_up_rect, self.scroll_down_rect = QRect(), QRect()
        else:
            self.scroll_up_rect, self.scroll_down_rect = QRect(), QRect()

        self._last_viewer_width = width
        self._last_viewer_height = self.height()
        self.is_layout_calculated = True # Đánh dấu là đã tính toán


    def resizeEvent(self, event):
        super().resizeEvent(event)
        # print(f"ReadingViewer resizeEvent: {self.size()}")
        # Gọi calculate_viewer_layout_rects trực tiếp hoặc qua timer ngắn
        # Nếu gọi trực tiếp, đảm bảo self.control_bar_frame và self.scroll_area đã có kích thước
        QTimer.singleShot(0, self.calculate_viewer_layout_rects)


    def showEvent(self, event):
        super().showEvent(event)
        # print("ReadingViewer showEvent")
        self.control_activated_button = False
        self.current_hovered_control = None
        self.dwell_start_time = None
        # Yêu cầu tính toán lại layout khi widget được hiển thị
        # Đảm bảo nó được gọi sau khi widget thực sự có kích thước
        QTimer.singleShot(10, self.calculate_viewer_layout_rects)


    def load_article(self, text):
        self.scroll_content_label.setText(text)
        self.scroll_area.verticalScrollBar().setValue(0)
        QTimer.singleShot(50, self.calculate_viewer_layout_rects)


    def update_gaze(self, x, y):
        self.gaze_x = x
        self.gaze_y = y


    def process_gaze_interactions_viewer(self):
        if not self.isVisible() or not self.is_layout_calculated: # Thêm kiểm tra is_layout_calculated
            return

        self.update() # Yêu cầu vẽ lại cho gaze point và dwell progress

        if self.gaze_x < 0 or self.gaze_y < 0:
            if self.current_hovered_control:
                self.current_hovered_control = None
                self.dwell_start_time = None
                self.control_activated_button = False
            return

        new_hovered_control = None
        # Tọa độ gaze (self.gaze_x, self.gaze_y) là local so với ReadingViewerWidget
        # back_button_rect cũng có tọa độ local so với ReadingViewerWidget
        if self.back_button_rect.isValid() and self.back_button_rect.contains(self.gaze_x, self.gaze_y):
            new_hovered_control = "BACK_BUTTON"
        elif self.scroll_up_rect.isValid() and self.scroll_up_rect.contains(self.gaze_x, self.gaze_y):
            new_hovered_control = "SCROLL_UP"
        elif self.scroll_down_rect.isValid() and self.scroll_down_rect.contains(self.gaze_x, self.gaze_y):
            new_hovered_control = "SCROLL_DOWN"

        if new_hovered_control != self.current_hovered_control:
            self.current_hovered_control = new_hovered_control
            self.dwell_start_time = time.perf_counter() if new_hovered_control else None
            self.control_activated_button = False

        if self.current_hovered_control and self.dwell_start_time:
            elapsed_time = time.perf_counter() - self.dwell_start_time
            is_scroll_action = self.current_hovered_control in ["SCROLL_UP", "SCROLL_DOWN"]
            is_button_action = self.current_hovered_control == "BACK_BUTTON"
            current_dwell_time_needed = self.dwell_threshold_scroll if is_scroll_action else self.dwell_threshold_button

            if elapsed_time >= current_dwell_time_needed:
                if is_button_action and not self.control_activated_button:
                    self.activate_viewer_control(self.current_hovered_control)
                    self.control_activated_button = True
                elif is_scroll_action:
                    self.activate_viewer_control(self.current_hovered_control)
                    self.dwell_start_time = time.perf_counter() - (current_dwell_time_needed * 0.7)


    def activate_viewer_control(self, control_name):
        if control_name == "BACK_BUTTON":
            print("DEBUG: Back to menu emitted from ReadingViewer") # Thêm debug
            self.back_to_menu.emit()
        elif control_name == "SCROLL_UP":
            current_val = self.scroll_area.verticalScrollBar().value()
            self.scroll_area.verticalScrollBar().setValue(current_val - self.scroll_speed)
        elif control_name == "SCROLL_DOWN":
            current_val = self.scroll_area.verticalScrollBar().value()
            self.scroll_area.verticalScrollBar().setValue(current_val + self.scroll_speed)


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Nền đã được set bằng palette

        # Vẽ nút "Quay lại Menu"
        if self.back_button_rect.isValid(): # Chỉ vẽ nếu rect hợp lệ
            is_hovered_back = (self.current_hovered_control == "BACK_BUTTON")
            
            path = QPainterPath()
            # Tọa độ của back_button_rect đã là local với ReadingViewerWidget
            # nhưng nó được định nghĩa để nằm trong control_bar_frame.
            # Chúng ta cần đảm bảo nó được vẽ đúng vị trí.
            # back_button_rect_to_draw là QRect(x, y, w, h) trong đó x, y là tọa độ
            # so với ReadingViewerWidget (góc trên trái của control_bar_frame là (0,0) so với control_bar_frame)
            # self.back_button_rect đã được tính toán vị trí tương đối với widget này (0,0)
            # nghĩa là nó đã bao gồm vị trí của control_bar_frame.
            # Tuy nhiên, cách tính self.back_button_rect trong calculate_viewer_layout_rects
            # là đặt nó ở self.control_bar_frame.y() + (chiều cao control_bar - chiều cao nút) / 2
            # Chúng ta cần tính lại `back_button_rect_visual` chỉ cho mục đích vẽ, nằm trong `control_bar_frame`
            
            # Lấy tọa độ của control_bar_frame so với ReadingViewerWidget
            # control_bar_origin = self.control_bar_frame.pos() # Đây là self.control_bar_frame.geometry().topLeft()
            
            # back_button_visual_rect = QRect(
            #     control_bar_origin.x() + self.back_button_rect.x(), # x của nút so với control_bar
            #     control_bar_origin.y() + self.back_button_rect.y(), # y của nút so với control_bar
            #     self.back_button_rect.width(),
            #     self.back_button_rect.height()
            # )
            # KHÔNG CẦN PHỨC TẠP NHƯ VẬY, self.back_button_rect đã là tọa độ trong ReadingViewerWidget
            # dựa trên chiều cao của control_bar_frame.

            path.addRoundedRect(QRectF(self.back_button_rect), 12, 12)

            base_back_color = QColor("#D32F2F")
            hover_back_color = base_back_color.lighter(140)
            
            painter.setBrush(hover_back_color if is_hovered_back else base_back_color)
            painter.setPen(QPen(base_back_color.darker(130), 2))
            painter.drawPath(path)

            painter.setPen(Qt.white)
            painter.setFont(self.control_button_font)
            painter.drawText(self.back_button_rect, Qt.AlignCenter, "❮ Menu")

            if is_hovered_back and self.dwell_start_time and not self.control_activated_button:
                elapsed = time.perf_counter() - self.dwell_start_time
                progress = min(1.0, elapsed / self.dwell_threshold_button)
                if progress < 1.0:
                    painter.setPen(QPen(QColor("#2ecc71"), 4))
                    arc_size = min(self.back_button_rect.width(), self.back_button_rect.height()) * 0.7 # Tăng kích thước vòng cung
                    if arc_size > 0 :
                        arc_rect_f = QRectF(0, 0, arc_size, arc_size)
                        arc_rect_f.moveCenter(self.back_button_rect.center()) # Dùng QRectF
                        start_angle = 90 * 16
                        span_angle = -int(progress * 360 * 16)
                        painter.drawArc(arc_rect_f, start_angle, span_angle)

        # Vẽ Gaze Point
        if self.gaze_x >= 0 and self.gaze_y >= 0:
            gaze_outer_color = QColor(70, 130, 180, 180)
            gaze_inner_color = QColor(240, 248, 255, 220)
            outer_radius = 11; inner_radius = 4
            painter.setPen(Qt.NoPen); painter.setBrush(gaze_outer_color)
            painter.drawEllipse(QPoint(self.gaze_x, self.gaze_y), outer_radius, outer_radius)
            painter.setBrush(gaze_inner_color)
            painter.drawEllipse(QPoint(self.gaze_x, self.gaze_y), inner_radius, inner_radius)

    def sizeHint(self):
        if self.parentWidget(): return self.parentWidget().size()
        primary_screen = QApplication.primaryScreen()
        if primary_screen:
            return primary_screen.size() * 0.9
        return QSize(800, 600)