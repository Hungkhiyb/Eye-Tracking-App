# # music_menu.py
# from PyQt5.QtWidgets import QWidget
# from PyQt5.QtCore import Qt, QRect, QTimer, pyqtSignal
# from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QPalette
# import time

# class MusicMenuWidget(QWidget):
#     back_to_home = pyqtSignal()
#     music_selected_for_play = pyqtSignal(int) # Tín hiệu khi chọn bài hát để PHÁT
#     stop_music_requested = pyqtSignal()      # Tín hiệu khi yêu cầu DỪNG nhạc

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setAutoFillBackground(True)
#         self.dwell_threshold = 1.5 # Giữ nguyên như các widget khác của bạn
#         self.gaze_x = -100
#         self.gaze_y = -100
#         self.current_zone = None
#         self.dwell_start_time = None
#         self.zone_activated = False
#         self.zones = []

#         self.music_track_names = [ # Tên các bài hát để hiển thị
#             "Bài hát 1: Giai điệu vui vẻ",
#             "Bài hát 2: Nhạc nền thư giãn",
#             "Bài hát 3: Piano cổ điển",
#             "Bài hát 4: Âm thanh tự nhiên"
#         ]
#         # Nút dừng nhạc sẽ là một zone riêng biệt

#         palette = self.palette()
#         palette.setColor(QPalette.Window, QColor("#1A1A1A"))
#         self.setPalette(palette)

#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update) # Kết nối với self.update để vẽ lại
#         self.timer.start(50)

#     def update_gaze(self, x, y):
#         self.gaze_x = x
#         self.gaze_y = y
#         # self.update() # Có thể gọi update ở đây hoặc để timer xử lý

#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.Antialiasing)
#         w, h = self.width(), self.height()
#         self.zones = []

#         # === TOP BAR (Quay về) ===
#         top_rect_height = 80
#         top_rect = QRect(0, 0, w, top_rect_height)
#         self.zones.append(("top_nav_music", top_rect)) # Đổi tên zone để tránh trùng
#         gradient_top = QLinearGradient(top_rect.topLeft(), top_rect.bottomLeft())
#         gradient_top.setColorAt(0, QColor("#2C3E50"))
#         gradient_top.setColorAt(1, QColor("#34495E"))
#         painter.setBrush(gradient_top)
#         painter.setPen(Qt.NoPen)
#         painter.drawRect(top_rect)
#         painter.setPen(QColor("#ECF0F1"))
#         font_top = QFont("Arial", 20, QFont.Bold)
#         painter.setFont(font_top)
#         painter.drawText(top_rect, Qt.AlignCenter, "⬅ Quay về trang chủ")

#         # === LƯỚI BÀI HÁT (Ví dụ: 2 cột, 2 hàng) + 1 ô dừng nhạc ===
#         grid_margin = 20
#         content_top_offset = top_rect_height + grid_margin
        
#         num_cols = 2
#         # Tổng số ô = số bài hát + 1 ô dừng nhạc
#         num_music_items = len(self.music_track_names)
#         total_items_to_display = num_music_items + 1 # Thêm ô dừng nhạc
        
#         # Ước tính chiều cao và chiều rộng cho mỗi ô
#         # Giả sử chúng ta muốn 3 hàng (2 hàng nhạc, 1 hàng nút dừng)
#         # Hoặc 2 hàng, hàng cuối có nút dừng
#         # Đơn giản: 2x2 cho nhạc, 1 ô bự cho dừng nhạc ở dưới hoặc bên cạnh

#         # Layout: 4 ô nhạc ở trên (2x2), 1 ô Dừng nhạc ở dưới cùng, chiếm toàn bộ chiều rộng
        
#         # Vùng cho 4 ô nhạc
#         music_grid_h = (h - content_top_offset - grid_margin * 2 - 80) // 2 # 80 là chiều cao cho nút dừng
#         cell_w_music = (w - (num_cols + 1) * grid_margin) // num_cols
#         cell_h_music = music_grid_h

#         for i in range(num_music_items):
#             row = i // num_cols
#             col = i % num_cols
            
#             x = grid_margin + col * (cell_w_music + grid_margin)
#             y = content_top_offset + row * (cell_h_music + grid_margin)
#             rect = QRect(x, y, cell_w_music, cell_h_music)
#             self.zones.append((f"musictrack_{i}", rect))

#             is_hovered = rect.contains(self.gaze_x, self.gaze_y)
#             base_color = QColor("#1ABC9C") # Xanh ngọc
#             hover_color = base_color.lighter(120)
            
#             painter.setBrush(hover_color if is_hovered else base_color)
#             painter.setPen(QPen(QColor("#16A085"), 2))
#             painter.drawRoundedRect(rect, 10, 10)

#             painter.setPen(Qt.white)
#             font_music = QFont("Arial", 14)
#             painter.setFont(font_music)
#             text_rect = rect.adjusted(10, 10, -10, -10)
#             painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, self.music_track_names[i])

#         # === Ô DỪNG NHẠC ===
#         stop_button_h = 70
#         stop_button_y = h - stop_button_h - grid_margin
#         stop_button_rect = QRect(grid_margin, stop_button_y, w - 2 * grid_margin, stop_button_h)
#         self.zones.append(("stop_music_button", stop_button_rect))

#         is_hovered_stop = stop_button_rect.contains(self.gaze_x, self.gaze_y)
#         stop_base_color = QColor("#E74C3C") # Đỏ
#         stop_hover_color = stop_base_color.lighter(120)

#         painter.setBrush(stop_hover_color if is_hovered_stop else stop_base_color)
#         painter.setPen(QPen(QColor("#C0392B"), 2))
#         painter.drawRoundedRect(stop_button_rect, 10, 10)

#         painter.setPen(Qt.white)
#         font_stop = QFont("Arial", 18, QFont.Bold)
#         painter.setFont(font_stop)
#         painter.drawText(stop_button_rect, Qt.AlignCenter, "Dừng phát nhạc")

#         # === ĐIỂM NHÌN ===
#         if self.gaze_x >= 0 and self.gaze_y >= 0:
#             painter.setBrush(QColor(255, 255, 255, 150))
#             painter.setPen(Qt.NoPen)
#             painter.drawEllipse(int(self.gaze_x) - 10, int(self.gaze_y) - 10, 20, 20)
#             # Dwell progress circle (nếu muốn thêm)
#             if self.current_zone is not None and not self.zone_activated and self.dwell_start_time is not None:
#                 dwell_time_elapsed = time.time() - self.dwell_start_time
#                 if dwell_time_elapsed < self.dwell_threshold:
#                     progress = dwell_time_elapsed / self.dwell_threshold
#                     painter.setPen(QPen(Qt.cyan, 3))
#                     painter.drawArc(
#                         int(self.gaze_x) - 15, int(self.gaze_y) - 15,
#                         30, 30, 0, int(progress * 360 * 16)
#                     )


#         self.check_gaze_zone() # Gọi sau khi tất cả zones đã được định nghĩa

#     def check_gaze_zone(self):
#         hit_zone_name = None
#         # hit_zone_rect = None # Không cần thiết nếu chỉ dùng tên
#         for name, rect in self.zones:
#             if rect.contains(self.gaze_x, self.gaze_y):
#                 hit_zone_name = name
#                 # hit_zone_rect = rect # Không cần thiết
#                 break

#         if hit_zone_name != self.current_zone:
#             self.current_zone = hit_zone_name
#             self.dwell_start_time = time.time() if hit_zone_name else None
#             self.zone_activated = False
#         elif hit_zone_name and not self.zone_activated and self.dwell_start_time:
#             if (time.time() - self.dwell_start_time) >= self.dwell_threshold:
#                 self.zone_activated = True
#                 self.activate_zone(hit_zone_name)
#                 # Sau khi kích hoạt, reset dwell_start_time để tránh kích hoạt lại ngay
#                 # self.dwell_start_time = time.time() # Hoặc None để chờ gaze rời đi
#                 # Hoặc để logic trong app.py xử lý việc reset zone_activated của widget con

#     def activate_zone(self, zone_name):
#         print(f"MusicMenu: Activating zone '{zone_name}'")
#         if zone_name == "top_nav_music":
#             self.back_to_home.emit()
#         elif zone_name == "stop_music_button":
#             self.stop_music_requested.emit()
#         elif zone_name.startswith("musictrack_"):
#             try:
#                 idx = int(zone_name.split("_")[1])
#                 if 0 <= idx < len(self.music_track_names):
#                     self.music_selected_for_play.emit(idx)
#             except (ValueError, IndexError) as e:
#                 print(f"Error parsing music track index from zone: {zone_name}, error: {e}")
        
#         # Quan trọng: Để cho app.py quản lý việc reset zone_activated của widget con
#         # sau khi action đã được xử lý, để tránh kích hoạt lặp lại quá nhanh.

#     def sizeHint(self):
#         return self.parent().size() if self.parent() else super().sizeHint()

# music_menu.py
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRect, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QPalette
import time

class MusicMenuWidget(QWidget):
    back_to_home = pyqtSignal()
    music_selected_for_play = pyqtSignal(int) # Index của bài hát được chọn để PHÁT
    toggle_play_pause_requested = pyqtSignal() # Tín hiệu yêu cầu Tạm dừng/Tiếp tục

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
            "Dangerously - Charlie Puth"
        ]
        self.is_music_playing_ui_state = False # Trạng thái UI cho nút Play/Pause

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#1A1A1A"))
        self.setPalette(palette)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_widget_ui) # Đổi tên để rõ ràng
        self.timer.start(50)

    def update_widget_ui(self):
        """Yêu cầu widget vẽ lại chính nó."""
        self.update()

    def update_gaze(self, x, y):
        self.gaze_x = x
        self.gaze_y = y

    def set_playing_indicator(self, is_playing):
        """Cập nhật trạng thái của nút Play/Pause dựa trên trạng thái thực của media player."""
        self.is_music_playing_ui_state = is_playing
        self.update() # Yêu cầu vẽ lại để cập nhật text nút

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        self.zones = [] # Reset zones mỗi lần vẽ

        # === Thiết kế layout 6 ô (ví dụ: 3 cột, 2 hàng) ===
        # Hàng 1: Bài hát 1, Bài hát 2, Quay về Home
        # Hàng 2: Bài hát 3, Bài hát 4, Tạm dừng/Phát

        num_cols = 3
        num_rows = 2
        margin = 20

        cell_width = (w - (num_cols + 1) * margin) // num_cols
        cell_height = (h - (num_rows + 1) * margin) // num_rows
        
        font_music = QFont("Arial", 14)
        font_action = QFont("Arial", 16, QFont.Bold)

        # Định nghĩa các ô và hành động
        # (Tên zone, Text hiển thị, màu cơ bản, màu hover, font)
        # Lưu ý: "musictrack_idx" là quy ước cho các ô bài hát
        zone_definitions = [
            ("musictrack_0", self.music_track_names[0], QColor("#1ABC9C"), QColor("#1ABC9C").lighter(120), font_music),
            ("musictrack_1", self.music_track_names[1], QColor("#1ABC9C"), QColor("#1ABC9C").lighter(120), font_music),
            ("nav_home_music", "⬅ Về Home", QColor("#3498DB"), QColor("#3498DB").lighter(120), font_action), # Xanh dương
            
            ("musictrack_2", self.music_track_names[2], QColor("#1ABC9C"), QColor("#1ABC9C").lighter(120), font_music),
            ("musictrack_3", self.music_track_names[3], QColor("#1ABC9C"), QColor("#1ABC9C").lighter(120), font_music),
            ("toggle_play_pause_music", "Tạm dừng" if self.is_music_playing_ui_state else "▶ Phát nhạc", QColor("#F39C12"), QColor("#F39C12").lighter(120), font_action) # Cam
        ]

        current_item_idx = 0
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                if current_item_idx >= len(zone_definitions):
                    break

                zone_name, text, base_color, hover_color, item_font = zone_definitions[current_item_idx]
                
                # Điều chỉnh text cho nút Play/Pause dựa trên trạng thái
                if zone_name == "toggle_play_pause_music":
                    text = "❚❚ Tạm dừng" if self.is_music_playing_ui_state else "▶ Phát nhạc"


                x = margin + col_idx * (cell_width + margin)
                y = margin + row_idx * (cell_height + margin)
                rect = QRect(x, y, cell_width, cell_height)
                self.zones.append((zone_name, rect))

                is_hovered = rect.contains(self.gaze_x, self.gaze_y)
                
                painter.setBrush(hover_color if is_hovered else base_color)
                painter.setPen(QPen(base_color.darker(120), 2)) # Viền đậm hơn
                painter.drawRoundedRect(rect, 15, 15)

                painter.setPen(Qt.white)
                painter.setFont(item_font)
                text_rect = rect.adjusted(10, 10, -10, -10) # Lề cho text
                painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, text)
                
                current_item_idx += 1
            if current_item_idx >= len(zone_definitions):
                    break


        # === ĐIỂM NHÌN ===
        if self.gaze_x >= 0 and self.gaze_y >= 0:
            painter.setBrush(QColor(255, 0, 0, 180)) # Màu đỏ cho điểm nhìn
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(self.gaze_x) - 8, int(self.gaze_y) - 8, 16, 16)
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(int(self.gaze_x) - 10, int(self.gaze_y) - 10, 20, 20)

            # Dwell progress circle (tùy chọn)
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

    def check_gaze_zone(self):
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
        elif zone_name.startswith("musictrack_"):
            try:
                idx = int(zone_name.split("_")[1])
                if 0 <= idx < len(self.music_track_names):
                    self.music_selected_for_play.emit(idx)
            except (ValueError, IndexError) as e:
                print(f"Error parsing music track index from zone: {zone_name}, error: {e}")
        
        # Để app.py quản lý việc reset zone_activated sau khi action được xử lý
        # self.zone_activated = False # Không reset ở đây ngay
        # self.current_zone = None
        # self.dwell_start_time = None

    def sizeHint(self):
        return self.parent().size() if self.parent() else super().sizeHint()