# music_menu.py
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import Qt, QRect, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QPalette, QPixmap
import time
import math # Thêm math để tính toán số trang

class MusicMenuWidget(QWidget):
    back_to_home = pyqtSignal()
    music_selected_for_play = pyqtSignal(int) # Index của bài hát được chọn để PHÁT (global index)
    toggle_play_pause_requested = pyqtSignal() # Tín hiệu yêu cầu Tạm dừng/Tiếp tục
    
    # Tín hiệu mới cho việc chuyển trang (nếu bạn muốn xử lý bên ngoài)
    # Hoặc chúng ta có thể xử lý chuyển trang ngay trong widget này
    # next_page_requested = pyqtSignal()
    # prev_page_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self.dwell_threshold = 1.5
        self.gaze_x = -100
        self.gaze_y = -100
        self.current_zone = None
        self.dwell_start_time = None
        self.zone_activated = False
        self.zones = []

        self.music_track_names = [
            "Let Her Go - Passenger",
            "Me And My Broken Heart - Rob Thomas",
            "Maps - Maroon 5",
            "Dangerously - Charlie Puth",
            "Counting Stars - OneRepublic",
            "Payphone - Maroon 5",
            "Sugar - Maroon 5"
            # Thêm nhiều bài hát nếu muốn
        ]
        self.is_music_playing_ui_state = False # Trạng thái UI cho nút Play/Pause
        self.current_playing_track_name = "Chưa có bài hát nào đang phát" # Tên bài hát đang phát
        self.current_media_progress = 0 # Từ 0.0 đến 1.0

        # Pagination
        self.current_page = 0
        self.tracks_per_page = 3
        self.total_pages = math.ceil(len(self.music_track_names) / self.tracks_per_page)


        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#1A1A1A")) # Màu nền chính
        self.setPalette(palette)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_widget_ui)
        self.timer.start(50)

    def update_widget_ui(self):
        self.update()

    def update_gaze(self, x, y):
        self.gaze_x = x
        self.gaze_y = y

    def set_playing_indicator(self, is_playing):
        self.is_music_playing_ui_state = is_playing
        self.update()

    def set_current_playing_track_info(self, track_name, progress=0.0):
        """Cập nhật tên bài hát đang phát và tiến trình (nếu có)"""
        self.current_playing_track_name = track_name if track_name else "Chưa có bài hát nào đang phát"
        self.current_media_progress = progress
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        self.zones = []
        margin = 30

        # === Load ảnh ===
        back_icon = QPixmap("assets/home_icon.png")
        left_icon = QPixmap("assets/left_arrow.png")
        right_icon = QPixmap("assets/right_arrow.png")
        music_icon = QPixmap("assets/music_icon.png")
        play_icon = QPixmap("assets/play_icon.png")
        pause_icon = QPixmap("assets/pause_icon.png")
        play_hover = QPixmap("assets/play_hover.png")
        pause_hover = QPixmap("assets/pause_hover.png")

        # === Nút quay về ===
        back_rect = QRect(0, 0, w, 150)
        self.zones.append(("nav_home_music", back_rect))
        painter.setBrush(QColor(66, 165, 245, 255))
        painter.setPen(Qt.NoPen)
        painter.drawRect(back_rect)

        is_hovered = back_rect.contains(self.gaze_x, self.gaze_y)
        if is_hovered:
            painter.setBrush(QColor(0, 0, 255, 255))
            painter.setPen(Qt.NoPen)
            painter.drawRect(back_rect)
        icon_size = 150
        icon_rect = QRect(
            back_rect.center().x() - icon_size // 2,
            back_rect.center().y() - icon_size // 2,
            icon_size, icon_size
        )
        painter.drawPixmap(icon_rect, back_icon.scaled(icon_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # === Nút chuyển trang trái ===
        left_rect = QRect(0, back_rect.bottom(), 150, h - back_rect.height())
        self.zones.append(("nav_prev_page_music", left_rect))
        if left_rect.contains(self.gaze_x, self.gaze_y):
            painter.setBrush(QColor(255, 255, 255, 40))
            painter.setPen(Qt.NoPen)
            painter.drawRect(left_rect)
        icon_l = QRect(left_rect.center().x() - 25, left_rect.center().y() - 25, 50, 50)
        painter.drawPixmap(icon_l, left_icon.scaled(icon_l.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # === Nút chuyển trang phải ===
        right_rect = QRect(w - left_rect.width(), back_rect.bottom(), left_rect.width(), h - back_rect.height())
        self.zones.append(("nav_next_page_music", right_rect))
        if right_rect.contains(self.gaze_x, self.gaze_y):
            painter.setBrush(QColor(255, 255, 255, 40))
            painter.setPen(Qt.NoPen)
            painter.drawRect(right_rect)
        icon_r = QRect(right_rect.center().x() - 25, right_rect.center().y() - 25, 50, 50)
        painter.drawPixmap(icon_r, right_icon.scaled(icon_r.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # === Vẽ các bài hát ===
        usable_w = w - left_rect.width() - right_rect.width()
        usable_h = h - back_rect.height()
        center_x = left_rect.width()
        content_top = back_rect.bottom() + margin
        item_w = (usable_w - 4 * margin) // 3
        item_h = int(usable_h * 0.6)

        font_music = QFont("Arial", 14)

        start_idx = self.current_page * self.tracks_per_page
        for i in range(3):
            idx = start_idx + i
            if idx >= len(self.music_track_names):
                break
            x = center_x + margin + i * (item_w + margin)
            rect = QRect(x, content_top, item_w, item_h)
            self.zones.append((f"musictrack_{idx}", rect))

            if rect.contains(self.gaze_x, self.gaze_y):
                painter.setBrush(QColor(255, 255, 255, 40))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(rect, 20, 20)

            icon_size = item_h // 2
            icon_rect = QRect(rect.center().x() - icon_size // 2, rect.top() + 10, icon_size, icon_size)
            painter.drawPixmap(icon_rect, music_icon.scaled(icon_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            text_rect = QRect(rect.left(), icon_rect.bottom() + 10, rect.width(), 40)
            painter.setFont(font_music)
            painter.setPen(Qt.white)
            painter.drawText(text_rect, Qt.AlignCenter, self.music_track_names[idx])

        # === Tên bài đang phát ===
        song_rect = QRect(center_x + margin, content_top + item_h + 30, usable_w - 2 * margin, 40)
        painter.setFont(QFont("Arial", 16, QFont.Bold))
        painter.setPen(Qt.white)
        painter.drawText(song_rect, Qt.AlignCenter, self.current_playing_track_name)

        # === Thanh tiến trình ===
        bar_h = 20
        bar_rect = QRect(song_rect.left(), song_rect.bottom() + 10, song_rect.width(), bar_h)
        painter.setBrush(QColor("#7f8c8d"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bar_rect, 10, 10)

        filled = int(bar_rect.width() * self.current_media_progress)
        fill_rect = QRect(bar_rect.left(), bar_rect.top(), filled, bar_h)
        painter.setBrush(QColor("#f39c12"))
        painter.drawRoundedRect(fill_rect, 10, 10)

        # === Nút play/pause ===
        btn_size = 200
        btn_rect = QRect(left_rect.width(), bar_rect.bottom() + 30, usable_w, btn_size)
        self.zones.append(("toggle_play_pause_music", btn_rect))

        hovered = btn_rect.contains(self.gaze_x, self.gaze_y)

        icon = (
            pause_hover if self.is_music_playing_ui_state and hovered
            else pause_icon if self.is_music_playing_ui_state
            else play_hover if hovered
            else play_icon
        )

        if hovered:
            painter.setBrush(QColor(255, 255, 255, 40))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(btn_rect, 20, 20)

        scaled_icon = icon.scaled(
            btn_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        x = btn_rect.x() + (btn_rect.width() - scaled_icon.width()) // 2
        y = btn_rect.y() + (btn_rect.height() - scaled_icon.height()) // 2
        painter.drawPixmap(x, y, scaled_icon)


        # === Gaze Point và Dwell vòng tròn ===
        if self.gaze_x >= 0 and self.gaze_y >= 0:
            painter.setBrush(QColor(255, 0, 0, 180)) 
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.gaze_x - 8, self.gaze_y - 8, 16, 16)
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self.gaze_x - 10, self.gaze_y - 10, 20, 20)

            if self.current_zone and not self.zone_activated and self.dwell_start_time:
                progress = (time.time() - self.dwell_start_time) / self.dwell_threshold
                if progress < 1:
                    painter.setPen(QPen(Qt.cyan, 3))
                    painter.drawArc(self.gaze_x - 15, self.gaze_y - 15, 30, 30, 0, int(progress * 360 * 16))

        self.check_gaze_zone()


    def check_gaze_zone(self): # Giữ nguyên logic này
        hit_zone_name = None
        for name, rect in self.zones:
            if rect.contains(self.gaze_x, self.gaze_y):
                hit_zone_name = name
                break

        if hit_zone_name != self.current_zone:
            self.current_zone = hit_zone_name
            self.dwell_start_time = time.time() if hit_zone_name else None
            self.zone_activated = False
        elif hit_zone_name and not self.zone_activated and self.dwell_start_time:
            if (time.time() - self.dwell_start_time) >= self.dwell_threshold:
                self.zone_activated = True
                self.activate_zone(hit_zone_name)

    def activate_zone(self, zone_name):
        print(f"MusicMenu: Activating zone '{zone_name}'")
        if zone_name == "nav_home_music":
            self.back_to_home.emit()
        elif zone_name == "toggle_play_pause_music":
            self.toggle_play_pause_requested.emit()
        elif zone_name == "nav_next_page_music":
            self.current_page = (self.current_page + 1) % self.total_pages
            self.update()
        elif zone_name == "nav_prev_page_music":
            self.current_page = (self.current_page - 1 + self.total_pages) % self.total_pages
            self.update()
        elif zone_name.startswith("musictrack_"):
            try:
                # Tên zone giờ là global index, ví dụ "musictrack_0", "musictrack_1", ...
                track_global_idx = int(zone_name.split("_")[1])
                if 0 <= track_global_idx < len(self.music_track_names):
                    self.music_selected_for_play.emit(track_global_idx)
            except (ValueError, IndexError) as e:
                print(f"Error parsing music track index from zone: {zone_name}, error: {e}")
        
        # Quan trọng: Reset trạng thái dwell để người dùng có thể kích hoạt lại
        # hoặc kích hoạt zone khác mà không cần nhìn ra ngoài rồi nhìn lại.
        # Việc này thường do widget cha (EyeTrackingControlApp) thực hiện sau khi
        # một action hoàn tất để tránh kích hoạt lặp lại ngay lập tức.
        # Tuy nhiên, với các nút chuyển trang nội bộ, có thể cần reset ở đây.
        # Để nhất quán, hãy để EyeTrackingControlApp quản lý việc reset này.
        # self.current_zone = None
        # self.zone_activated = False
        # self.dwell_start_time = None


    def sizeHint(self): # Giữ nguyên
        return self.parent().size() if self.parent() else super().sizeHint()