# music_menu.py
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import Qt, QRect, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QPalette
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
        self.zones = [] # Reset zones mỗi lần vẽ
        
        margin = 20
        
        # --- 1. Nút Back to Home (Trên cùng, kéo dài hết chiều ngang) ---
        top_bar_height = int(h * 0.1)
        back_button_rect = QRect(0, 0, w, top_bar_height)
        self.zones.append(("nav_home_music", back_button_rect))
        
        # Vẽ nút Back
        is_hovered_back = back_button_rect.contains(self.gaze_x, self.gaze_y)
        back_color = QColor("#3498DB").lighter(120) if is_hovered_back else QColor("#3498DB")
        painter.fillRect(back_button_rect, back_color)
        painter.setPen(QPen(back_color.darker(120), 2))
        painter.drawRect(back_button_rect)
        painter.setPen(Qt.white)
        font_back = QFont("Arial", 16, QFont.Bold)
        painter.setFont(font_back)
        painter.drawText(back_button_rect, Qt.AlignCenter, "⬅ Về Trang Chủ")

        content_y_start = top_bar_height + margin

        # --- 2. Nút Chuyển Trang (Hai bên, kéo dài hết chiều cao còn lại) ---
        side_button_width = int(w * 0.2) # Giảm chiều rộng nút bên
        side_button_height = h - content_y_start
        
        # Nút Previous Page (Bên trái)
        prev_page_rect = QRect(0, content_y_start, side_button_width, side_button_height)
        if self.current_page > 0 : # Chỉ thêm zone nếu có thể quay lại
            self.zones.append(("nav_prev_page_music", prev_page_rect))
        
        is_hovered_prev = prev_page_rect.contains(self.gaze_x, self.gaze_y) and self.current_page > 0
        prev_page_color = QColor("#2ECC71").lighter(120) if is_hovered_prev else QColor("#2ECC71")
        if self.current_page == 0: # Làm mờ nếu không thể nhấn
            prev_page_color = QColor("#7f8c8d") # Màu xám
        painter.fillRect(prev_page_rect, prev_page_color)
        painter.setPen(QPen(prev_page_color.darker(120), 2))
        painter.drawRect(prev_page_rect)
        painter.setPen(Qt.white)
        painter.setFont(font_back) # Dùng font to
        painter.drawText(prev_page_rect, Qt.AlignCenter, "❮\nTrang\nTrước")

        # Nút Next Page (Bên phải)
        next_page_rect = QRect(w - side_button_width, content_y_start, side_button_width, side_button_height)
        if self.current_page < self.total_pages - 1: # Chỉ thêm zone nếu có thể tiến
            self.zones.append(("nav_next_page_music", next_page_rect))

        is_hovered_next = next_page_rect.contains(self.gaze_x, self.gaze_y) and self.current_page < self.total_pages - 1
        next_page_color = QColor("#2ECC71").lighter(120) if is_hovered_next else QColor("#2ECC71")
        if self.current_page >= self.total_pages - 1: # Làm mờ nếu không thể nhấn
             next_page_color = QColor("#7f8c8d") # Màu xám
        painter.fillRect(next_page_rect, next_page_color)
        painter.setPen(QPen(next_page_color.darker(120), 2))
        painter.drawRect(next_page_rect)
        painter.setPen(Qt.white)
        painter.setFont(font_back) # Dùng font to
        painter.drawText(next_page_rect, Qt.AlignCenter, "❯\nTrang\nSau")


        # --- Khu vực nội dung chính (ở giữa) ---
        central_content_x = side_button_width
        central_content_width = w - 2 * side_button_width
        
        # --- 3. Ba ô bài nhạc (ở trên khu vực trung tâm) ---
        track_button_y = content_y_start
        track_button_area_height = int(h * 0.35) # Tăng chiều cao khu vực này
        
        num_display_tracks = self.tracks_per_page
        track_button_width = (central_content_width - (num_display_tracks + 1) * margin) // num_display_tracks
        track_button_height = track_button_area_height - margin # Để có khoảng trống

        font_music = QFont("Arial", 14)
        
        start_track_idx = self.current_page * self.tracks_per_page
        for i in range(num_display_tracks):
            actual_track_idx = start_track_idx + i
            if actual_track_idx >= len(self.music_track_names):
                break # Không còn bài hát để hiển thị trên trang này

            track_name = self.music_track_names[actual_track_idx]
            track_x = central_content_x + margin + i * (track_button_width + margin)
            track_rect = QRect(track_x, track_button_y, track_button_width, track_button_height)
            self.zones.append((f"musictrack_{actual_track_idx}", track_rect)) # Sử dụng global index

            is_hovered_track = track_rect.contains(self.gaze_x, self.gaze_y)
            track_color_base = QColor("#1ABC9C")
            track_color = track_color_base.lighter(120) if is_hovered_track else track_color_base
            
            painter.setBrush(track_color)
            painter.setPen(QPen(track_color_base.darker(120), 2))
            painter.drawRoundedRect(track_rect, 10, 10)
            painter.setPen(Qt.white)
            painter.setFont(font_music)
            painter.drawText(track_rect.adjusted(5,5,-5,-5), Qt.AlignCenter | Qt.TextWordWrap, track_name)

        # --- 4. Tên bài hát đang phát (dưới 3 ô bài nhạc) ---
        current_song_label_y = track_button_y + track_button_area_height + margin // 2
        current_song_label_height = int(h * 0.1)
        current_song_rect = QRect(central_content_x, current_song_label_y, 
                                  central_content_width, current_song_label_height)
        painter.setPen(QColor("#ECF0F1")) # Màu chữ sáng
        font_current_song = QFont("Arial", 16, QFont.Bold)
        painter.setFont(font_current_song)
        painter.drawText(current_song_rect, Qt.AlignCenter, f"Đang phát: {self.current_playing_track_name}")

        # --- 5. Thanh tiến trình (dưới tên bài hát) ---
        progress_bar_y = current_song_label_y + current_song_label_height 
        progress_bar_height = int(h * 0.05) # Thanh mỏng hơn
        progress_bar_rect_outer = QRect(central_content_x + margin, progress_bar_y, 
                                    central_content_width - 2 * margin, progress_bar_height)
        
        # Vẽ nền thanh tiến trình
        painter.setBrush(QColor("#7F8C8D")) # Xám đậm
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(progress_bar_rect_outer, 5, 5)

        # Vẽ phần tiến trình đã chạy
        # (Giả sử self.current_media_progress là từ 0.0 đến 1.0)
        filled_width = int(progress_bar_rect_outer.width() * self.current_media_progress)
        progress_bar_rect_inner = QRect(progress_bar_rect_outer.x(), progress_bar_rect_outer.y(),
                                    filled_width, progress_bar_rect_outer.height())
        painter.setBrush(QColor("#F39C12")) # Màu cam cho tiến trình
        painter.drawRoundedRect(progress_bar_rect_inner, 5, 5)


        # --- 6. Nút Pause/Play (Chính giữa, dưới thanh tiến trình) ---
        play_pause_y = progress_bar_y + progress_bar_height + margin
        play_pause_button_height = int(h * 0.15) # To hơn chút
        play_pause_button_width = int(central_content_width * 0.4)
        play_pause_x = central_content_x + (central_content_width - play_pause_button_width) // 2
        play_pause_rect = QRect(play_pause_x, play_pause_y, play_pause_button_width, play_pause_button_height)
        self.zones.append(("toggle_play_pause_music", play_pause_rect))

        is_hovered_play_pause = play_pause_rect.contains(self.gaze_x, self.gaze_y)
        play_pause_text = "❚❚ Tạm dừng" if self.is_music_playing_ui_state else "▶ Phát nhạc"
        play_pause_color_base = QColor("#E67E22") # Cam đậm hơn
        play_pause_color = play_pause_color_base.lighter(120) if is_hovered_play_pause else play_pause_color_base
        
        painter.setBrush(play_pause_color)
        painter.setPen(QPen(play_pause_color_base.darker(120), 2))
        painter.drawRoundedRect(play_pause_rect, 15, 15)
        painter.setPen(Qt.white)
        font_play_pause = QFont("Arial", 18, QFont.Bold)
        painter.setFont(font_play_pause)
        painter.drawText(play_pause_rect, Qt.AlignCenter, play_pause_text)
        
        # --- Thông tin trang hiện tại ---
        page_info_y = h - margin - 20 # Gần cuối màn hình
        page_info_rect = QRect(central_content_x, page_info_y, central_content_width, 20)
        painter.setPen(QColor("#BDC3C7")) # Xám nhạt
        font_page_info = QFont("Arial", 12)
        painter.setFont(font_page_info)
        painter.drawText(page_info_rect, Qt.AlignCenter, f"Trang {self.current_page + 1} / {self.total_pages}")


        # === ĐIỂM NHÌN và DWELL PROGRESS === (Giữ nguyên từ code gốc của bạn)
        if self.gaze_x >= 0 and self.gaze_y >= 0:
            painter.setBrush(QColor(255, 0, 0, 180)) 
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(self.gaze_x) - 8, int(self.gaze_y) - 8, 16, 16)
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(int(self.gaze_x) - 10, int(self.gaze_y) - 10, 20, 20)

            if self.current_zone is not None and not self.zone_activated and self.dwell_start_time is not None:
                dwell_time_elapsed = time.time() - self.dwell_start_time
                if dwell_time_elapsed < self.dwell_threshold:
                    progress = dwell_time_elapsed / self.dwell_threshold
                    painter.setPen(QPen(Qt.cyan, 3))
                    painter.drawArc(
                        int(self.gaze_x) - 15, int(self.gaze_y) - 15,
                        30, 30, 0, int(progress * 360 * 16)
                    )
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
            if self.current_page < self.total_pages - 1:
                self.current_page += 1
                self.update() # Yêu cầu vẽ lại để hiển thị trang mới
                # self.next_page_requested.emit() # Nếu muốn xử lý bên ngoài
        elif zone_name == "nav_prev_page_music":
            if self.current_page > 0:
                self.current_page -= 1
                self.update() # Yêu cầu vẽ lại
                # self.prev_page_requested.emit() # Nếu muốn xử lý bên ngoài
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