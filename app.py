import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QLabel, QPushButton, QComboBox, QSlider, QGroupBox, QHBoxLayout, QMessageBox, QStackedLayout, QGraphicsOpacityEffect)
from PyQt5.QtCore import Qt, QTimer, QRect, QUrl, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPalette, QColor, QImage, QPixmap, QPainter, QPen, QLinearGradient
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import cv2
import numpy as np
import torch
import mediapipe as mp
from train import GazeDualHeadMLP
from joblib import load
import pyautogui
import math
import time
from reading_menu import ReadingMenuWidget
from reading_viewer import ReadingViewerWidget

from music_menu import MusicMenuWidget

class CustomNotificationWidget(QLabel): # Đổi tên thành tên chung hơn
    def __init__(self, text, parent=None, duration=2500, bg_color=QColor(0, 150, 200, 220)): # Thêm bg_color
        super().__init__(text, parent)
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: rgba({bg_color.red()}, {bg_color.green()}, {bg_color.blue()}, {bg_color.alpha()});
                color: white;
                font-size: 26px; /* Điều chỉnh font size nếu cần */
                font-weight: bold;
                padding: 20px; /* Padding hợp lý */
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 100); /* Viền trắng mờ */
            }}
        """)
        self.setAlignment(Qt.AlignCenter)
        self.adjustSize() # Điều chỉnh kích thước theo nội dung

        # Hiệu ứng Opacity
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)

        # Tự động đóng sau một khoảng thời gian
        QTimer.singleShot(duration, self.close_smoothly)

    def show_centered_with_animation(self): # Đổi tên hàm
        if self.parentWidget():
            parent_rect = self.parentWidget().geometry()
            self.move(parent_rect.center() - self.rect().center())
        
        self.animation_in = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation_in.setDuration(300) # Thời gian hiện ra
        self.animation_in.setStartValue(0.0)
        self.animation_in.setEndValue(1.0) # Độ mờ cuối cùng
        self.animation_in.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation_in.start()
        self.show() # Hiển thị widget

    def close_smoothly(self):
        self.animation_out = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation_out.setDuration(500) # Thời gian mờ đi
        self.animation_out.setStartValue(1.0)
        self.animation_out.setEndValue(0.0)
        self.animation_out.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation_out.finished.connect(self.deleteLater) # Xóa widget sau khi animation kết thúc
        self.animation_out.start()
class EyeTrackingControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracking Control Interface")

        self.init_eye_tracking()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            sys.exit()
        self.active_custom_notification_ref = None
        

        # Setup main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.stack = QStackedLayout()
        self.central_widget.setLayout(self.stack)
        self.main_control_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_control_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        self.stack.addWidget(self.main_control_widget)

        # Setup reading menu
        self.reading_menu = ReadingMenuWidget()
        self.reading_menu.article_selected.connect(self.open_article)
        self.reading_menu.back_to_home.connect(self.go_home)
        self.reading_menu.turn_page.connect(self.turn_page)
        self.stack.addWidget(self.reading_menu)

        # Setup reading viewer
        self.reading_viewer = ReadingViewerWidget()
        self.reading_viewer.back_to_menu.connect(self.open_reading_menu)
        self.stack.addWidget(self.reading_viewer)

        # self.music_menu_widget = MusicMenuWidget() # Đổi tên biến để rõ ràng
        # self.music_menu_widget.back_to_home.connect(self.go_home)
        # self.music_menu_widget.music_selected_for_play.connect(self.handle_music_selection) # Kết nối với hàm xử lý chọn nhạc
        # self.music_menu_widget.stop_music_requested.connect(self.stop_current_music)      # Kết nối với hàm dừng nhạc
        # self.stack.addWidget(self.music_menu_widget)
        # <<<<< SETUP MUSIC MENU WIDGET >>>>>
        self.music_menu_widget = MusicMenuWidget()
        self.music_menu_widget.back_to_home.connect(self.go_home)
        self.music_menu_widget.music_selected_for_play.connect(self.handle_music_selection)
        # self.music_menu_widget.stop_music_requested.connect(self.stop_current_music) # Tín hiệu này không còn nữa
        self.music_menu_widget.toggle_play_pause_requested.connect(self.handle_toggle_play_pause) # <<<<< KẾT NỐI TÍN HIỆU MỚI
        self.stack.addWidget(self.music_menu_widget)

        # <<<<< MEDIAPLAYER SETUP >>>>>
        self.media_player = QMediaPlayer(None, QMediaPlayer.StreamPlayback)
        self.media_player.setVolume(70) # Hoặc âm lượng bạn muốn
        # QUAN TRỌNG: Thay thế bằng đường dẫn thực tế đến các file nhạc của bạn
        self.music_files_paths = [
            "Let Her Go.mp3",
            "Me And My Broken Heart.mp3",
            "Maps.mp3",
            "Dangerously.mp3"
        ]
        self.current_playing_music_idx = -1 # Index của bài hát đang phát

        # Kiểm tra sự tồn tại của file nhạc (tùy chọn nhưng nên có)
        for i, f_path in enumerate(self.music_files_paths):
            if not os.path.exists(f_path):
                print(f"Cảnh báo: File nhạc '{f_path}' (cho bài {i+1} trong MusicMenuWidget) không tồn tại.")

        self.current_playing_music_index = -1
        self.is_music_paused_by_user = False
        
        
        # Setup top row for three zones
        self.top_row = QWidget()
        self.top_layout = QHBoxLayout(self.top_row)
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        self.top_layout.setSpacing(10)

        # Setup bottom row for one large zone
        self.bottom_row = QWidget()
        self.bottom_layout = QHBoxLayout(self.bottom_row)
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Setup debug overlay panel (small and transparent in the corner)
        self.debug_overlay = QWidget(self)
        self.debug_overlay.setGeometry(10, 10, 300, 100)
        self.debug_layout = QVBoxLayout(self.debug_overlay)

        self.video_label = QLabel()
        self.video_label.setFixedSize(200, 150)

        self.debug_layout.addWidget(self.video_label)
        self.debug_label = QLabel("Eye Tracking Debug Info")
        self.debug_label.setStyleSheet("""
            font-size: 12px; 
            color: #eeeeee; 
            background-color: rgba(30, 30, 30, 200); 
            padding: 4px;
            border-radius: 6px;
        """)
        self.debug_layout.addWidget(self.debug_label)

        # Add rows to main layout
        self.main_layout.addWidget(self.top_row, 3)    # Top row gets 3/4 of vertical space
        self.main_layout.addWidget(self.bottom_row, 1) # Bottom row gets 1/4 of vertical space

        # Define interaction zones (will be visualized in the paintEvent)
        self.setup_interaction_zones()

        # Current eye gaze point
        self.gaze_x = 0
        self.gaze_y = 0

        # Dwell time tracking
        self.current_zone = None
        self.dwell_start_time = 0
        self.dwell_threshold = 1.5  # seconds
        self.zone_activated = False

        # Setup timer for camera updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms (approx. 33 fps)

        # Setup UI refresh timer
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update)
        self.ui_timer.start(50)  # Update UI every 50ms

        # Add ESC key to exit fullscreen
        self.debug_overlay.raise_()
        self.showFullScreen()

        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
                color: white;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #444;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #333;
            }
            QPushButton:pressed {
                background-color: #555;
            }
        """)

    def keyPressEvent(self, event):
        """Handle key press events"""
        # ESC key to exit fullscreen
        if event.key() == Qt.Key_Escape:
            self.close()

    def init_eye_tracking(self):
        # Load models and initialize tracking
        self.pred_queue = []
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = pyautogui.size()
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
        
        # Load scalers and model
        self.scaler_input = load('scaler_input.joblib')
        self.scaler_target = load('scaler_target.joblib')
        
        input_size = self.scaler_input.n_features_in_
        self.model = GazeDualHeadMLP(input_size=input_size)
        self.model.load_state_dict(torch.load('gaze_model.pth', map_location='cpu'))
        self.model.eval()
        
        # MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)

        # ===== CẤU HÌNH CAMERA (phải khớp với feature_extraction.py) =====
        self.CAM_MTX = np.array([[950, 0, 640],
                            [0, 950, 360],
                            [0, 0, 1]], dtype=np.float64)
        self.DIST_COEFFS = np.zeros((4, 1))

        # ===== MODEL POINTS (3D) =====
        self.MODEL_POINTS = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)
        
        # Landmark IDs (same as in your infer.py)
        self.LANDMARK_IDS = {
            'nose': 1,
            'chin': 152,
            'left_eye': 33,
            'right_eye': 263,
            'mouth_left': 61,
            'mouth_right': 291,
            'left_iris': 468,
            'right_iris': 473,
            'left_inner': 133,
            'left_outer': 33,
            'right_inner': 362,
            'right_outer': 263,
            'left_eye_top': 159,
            'left_eye_bottom': 145,
            'right_eye_top': 386,
            'right_eye_bottom': 374,
            'forehead': 10
        }

        self.temporal_state = {
            "prev_iris_pos": None,
            "prev_velocity": 0.0
        }
        
    def setup_interaction_zones(self):
        """Define the interaction zones and their actions"""
        # Define zones - format: (name, action_callback, color)
        self.zones = [
            
            ("Gọi người chăm sóc", self.call_caretaker, QColor("#ef5350")),  # đỏ nhạt
            ("Bật nhạc", self.show_music_menu_action, QColor("#66bb6a")), # Dòng mới
            ("Đọc sách", self.read_book, QColor("#42a5f5")),                 # xanh dương
            ("SOS", self.emergency, QColor("#d32f2f")),                      # đỏ đậm
            ("", None, QColor("#424242")),                                   # xám
            ("", None, QColor("#424242")),                                   # xám
        ]

        
        # Zone rectangles will be calculated during paintEvent based on window size
        
    def paintEvent(self, event):
        painter = QPainter(self)

        # Get window dimensions
        window_width = self.width()
        window_height = self.height()

        # Zone size
        zone_width = (window_width - 40) // 3
        zone_height = (window_height - 40) // 2

        self.zone_rects = []

        for i, (name, _, color) in enumerate(self.zones):
            row = i // 3
            col = i % 3

            x = 10 + col * (zone_width + 10)
            y = 10 + row * (zone_height + 10)

            rect = QRect(x, y, zone_width, zone_height)
            self.zone_rects.append(rect)

            # Gradient fill
            gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
            gradient.setColorAt(0, color.lighter(110))
            gradient.setColorAt(1, color.darker(110))
            painter.fillRect(rect, gradient)

            # Draw base border
            painter.setPen(QPen(Qt.black, 2))
            painter.drawRect(rect)

            # Highlight border if dwelling
            if i == self.current_zone:
                dwell_time = time.time() - self.dwell_start_time
                if not self.zone_activated:
                    progress = dwell_time / self.dwell_threshold
                    pen_width = 4 + int(progress * 4)
                    painter.setPen(QPen(Qt.cyan, pen_width))
                else:
                    painter.setPen(QPen(Qt.green, 6))
                painter.drawRect(rect)

            # Draw name
            if name:
                painter.setPen(Qt.white)
                font = QFont()
                font.setPointSize(24)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(rect, Qt.AlignCenter, name)

        # Draw gaze point
        if hasattr(self, 'screen_gaze_x') and hasattr(self, 'screen_gaze_y'):
            # Inner red circle
            painter.setBrush(QColor(255, 0, 0, 180))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(self.screen_gaze_x) - 8, int(self.screen_gaze_y) - 8, 16, 16)

            # Outer white ring
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(int(self.screen_gaze_x) - 10, int(self.screen_gaze_y) - 10, 20, 20)

            # Draw dwell progress circle
            if self.current_zone is not None and not self.zone_activated:
                dwell_time = time.time() - self.dwell_start_time
                if dwell_time < self.dwell_threshold:
                    progress = dwell_time / self.dwell_threshold
                    painter.setPen(QPen(Qt.cyan, 3))
                    painter.drawArc(
                        int(self.screen_gaze_x) - 15,
                        int(self.screen_gaze_y) - 15,
                        30, 30,
                        0,
                        int(progress * 360 * 16)
                    )

        
    def get_point(self, face_landmarks, idx, frame_shape):
        h, w = frame_shape[:2]
        lm = face_landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h])
        
    def get_iris_features(self, face_landmarks, frame_shape):
        def norm_pos(iris, inner, outer):
            d = outer - inner
            norm = (iris - inner) / np.linalg.norm(d) if np.linalg.norm(d) > 0 else [0, 0]
            return norm
        
        left_iris = self.get_point(face_landmarks, self.LANDMARK_IDS['left_iris'], frame_shape)
        left_inner = self.get_point(face_landmarks, self.LANDMARK_IDS['left_inner'], frame_shape)
        left_outer = self.get_point(face_landmarks, self.LANDMARK_IDS['left_outer'], frame_shape)
        
        right_iris = self.get_point(face_landmarks, self.LANDMARK_IDS['right_iris'], frame_shape)
        right_inner = self.get_point(face_landmarks, self.LANDMARK_IDS['right_inner'], frame_shape)
        right_outer = self.get_point(face_landmarks, self.LANDMARK_IDS['right_outer'], frame_shape)
        
        norm_left = norm_pos(left_iris, left_inner, left_outer)
        norm_right = norm_pos(right_iris, right_inner, right_outer)
        
        return norm_left.tolist() + norm_right.tolist()
    
    def get_ear(self, face_landmarks, frame_shape):
        def compute(top, bottom, outer, inner):
            top_pt = self.get_point(face_landmarks, top, frame_shape)
            bottom_pt = self.get_point(face_landmarks, bottom, frame_shape)
            outer_pt = self.get_point(face_landmarks, outer, frame_shape)
            inner_pt = self.get_point(face_landmarks, inner, frame_shape)
            
            v = np.linalg.norm(top_pt - bottom_pt)
            h = np.linalg.norm(outer_pt - inner_pt)
            return v / h if h != 0 else 0
        
        ear_l = compute(self.LANDMARK_IDS['left_eye_top'], self.LANDMARK_IDS['left_eye_bottom'], 
                        self.LANDMARK_IDS['left_outer'], self.LANDMARK_IDS['left_inner'])
        ear_r = compute(self.LANDMARK_IDS['right_eye_top'], self.LANDMARK_IDS['right_eye_bottom'], 
                        self.LANDMARK_IDS['right_outer'], self.LANDMARK_IDS['right_inner'])
        return ear_l, ear_r

    def get_lid_ratio(self, face_landmarks, frame_shape):
        forehead = self.get_point(face_landmarks, self.LANDMARK_IDS['forehead'], frame_shape)
        chin = self.get_point(face_landmarks, self.LANDMARK_IDS['chin'], frame_shape)
        face_h = abs(chin[1] - forehead[1])
        
        if face_h == 0: 
            return 0, 0
        
        left_top = self.get_point(face_landmarks, self.LANDMARK_IDS['left_eye_top'], frame_shape)
        left_bottom = self.get_point(face_landmarks, self.LANDMARK_IDS['left_eye_bottom'], frame_shape)
        left = abs(left_top[1] - left_bottom[1]) / face_h
        
        right_top = self.get_point(face_landmarks, self.LANDMARK_IDS['right_eye_top'], frame_shape)
        right_bottom = self.get_point(face_landmarks, self.LANDMARK_IDS['right_eye_bottom'], frame_shape)
        right = abs(right_top[1] - right_bottom[1]) / face_h
        
        return left, right

    def get_head_pose(self, face_landmarks, frame_shape):
        try:
            image_points = np.array([
                self.get_point(face_landmarks, self.LANDMARK_IDS['nose'], frame_shape),
                self.get_point(face_landmarks, self.LANDMARK_IDS['chin'], frame_shape),
                self.get_point(face_landmarks, self.LANDMARK_IDS['left_eye'], frame_shape),
                self.get_point(face_landmarks, self.LANDMARK_IDS['right_eye'], frame_shape),
                self.get_point(face_landmarks, self.LANDMARK_IDS['mouth_left'], frame_shape),
                self.get_point(face_landmarks, self.LANDMARK_IDS['mouth_right'], frame_shape),
            ], dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(
                self.MODEL_POINTS, image_points, self.CAM_MTX, self.DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                R, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
                singular = sy < 1e-6
                
                if not singular:
                    pitch = math.atan2(R[2, 1], R[2, 2])
                    yaw = math.atan2(-R[2, 0], sy)
                    roll = math.atan2(R[1, 0], R[0, 0])
                else:
                    pitch = math.atan2(-R[1, 2], R[1, 1])
                    yaw = math.atan2(-R[2, 0], sy)
                    roll = 0
                    
                return {
                    'pitch': math.degrees(pitch),
                    'yaw': math.degrees(yaw),
                    'roll': math.degrees(roll),
                    'trans_x': float(tvec[0]),
                    'trans_y': float(tvec[1]),
                    'trans_z': float(tvec[2])
                }
        except:
            pass
        
        return {
            'pitch': 0.0,
            'yaw': 0.0,
            'roll': 0.0,
            'trans_x': 0.0,
            'trans_y': 0.0,
            'trans_z': 0.0
        }
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                features = []
                
                # Landmark positions
                for idx in range(478):
                    lm = face_landmarks.landmark[idx]
                    features.append(lm.x * self.CAMERA_WIDTH)
                    features.append(lm.y * self.CAMERA_HEIGHT)
                
                # Eye features
                eye_top = face_landmarks.landmark[159].y * self.CAMERA_HEIGHT
                eye_bottom = face_landmarks.landmark[145].y * self.CAMERA_HEIGHT
                iris_center = face_landmarks.landmark[468].y * self.CAMERA_HEIGHT
                
                eye_vert_dist = eye_bottom - eye_top
                iris_ratio = (iris_center - eye_top) / eye_vert_dist if eye_vert_dist != 0 else 0.5
                features.append(eye_vert_dist)
                features.append(iris_ratio)
                
                # Iris features
                iris_features = self.get_iris_features(face_landmarks, frame.shape)
                features.extend(iris_features)
                
                # EAR (2 features)
                ear_l, ear_r = self.get_ear(face_landmarks, frame.shape)
                features.append(ear_l)
                features.append(ear_r)
                
                # Lid ratio (2 features)
                lid_l, lid_r = self.get_lid_ratio(face_landmarks, frame.shape)
                features.append(lid_l)
                features.append(lid_r)
                
                # Temporal features (velocity, acceleration)
                current_iris_pos = self.get_point(face_landmarks, self.LANDMARK_IDS['left_iris'], frame.shape)
                prev_iris_pos = self.temporal_state["prev_iris_pos"]
                prev_velocity = self.temporal_state["prev_velocity"]

                if prev_iris_pos is not None:
                    velocity = np.linalg.norm(current_iris_pos - prev_iris_pos)
                    acceleration = velocity - prev_velocity
                else:
                    velocity = 0.0
                    acceleration = 0.0

                self.temporal_state["prev_velocity"] = velocity
                self.temporal_state["prev_iris_pos"] = current_iris_pos
                features.append(velocity)
                features.append(acceleration)
                
                # Illumination (placeholder)
                features.append(127)  # illum_mean
                features.append(20)   # illum_std
                
                # Head pose
                head_pose = self.get_head_pose(face_landmarks, frame.shape)
                features.append(head_pose['pitch'])
                features.append(head_pose['yaw'])
                features.append(head_pose['roll'])
                features.append(head_pose['trans_x'])
                features.append(head_pose['trans_y'])
                features.append(head_pose['trans_z'])
                
                # Prediction
                input_data = np.array(features).reshape(1, -1)
                input_scaled = self.scaler_input.transform(input_data)
                
                with torch.no_grad():
                    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
                    pred = self.model(input_tensor).numpy()
                    pred = self.scaler_target.inverse_transform(pred)[0]
                
                x_pred = int(np.clip(pred[0], 0, self.SCREEN_WIDTH))
                y_pred = int(np.clip(pred[1], 0, self.SCREEN_HEIGHT))
                
                self.pred_queue.append((x_pred, y_pred))
                if len(self.pred_queue) > 5:
                    self.pred_queue.pop(0)
                
                self.gaze_x = int(np.mean([p[0] for p in self.pred_queue]))
                self.gaze_y = int(np.mean([p[1] for p in self.pred_queue]))
                
                # Draw gaze point on camera frame
                cv2.circle(frame, (int(self.gaze_x), int(self.gaze_y)), 5, (0, 0, 255), -1)
                
                # Convert frame to RGB for Qt
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                
                # Convert to QImage and display
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    200, 150, Qt.KeepAspectRatio))
                
                # Map the gaze coordinates from the camera frame to screen coordinates
                self.map_gaze_to_screen()
                
                # Check if gaze is in an interaction zone
                if self.stack.currentWidget() == self.main_control_widget:
                    self.check_zone_interaction()

                
                # Update debug info
                # self.update_debug_info()

                # Truyền gaze point vào giao diện đang hiển thị (nếu có xử lý)
                current_widget = self.stack.currentWidget()
                if isinstance(current_widget, ReadingMenuWidget):
                    current_widget.update_gaze(self.screen_gaze_x, self.screen_gaze_y)
                elif isinstance(current_widget, MusicMenuWidget):
                    current_widget.update_gaze(self.screen_gaze_x, self.screen_gaze_y)
                elif isinstance(current_widget, ReadingViewerWidget):
                    current_widget.update_gaze(self.screen_gaze_x, self.screen_gaze_y)


                
    def map_gaze_to_screen(self):
        """Map gaze coordinates from camera frame to screen coordinates"""
        # Get window dimensions
        window_width = self.width()
        window_height = self.height()
        
        # Get camera frame dimensions
        frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Simple linear mapping (in a real implementation, you would need proper calibration)
        self.screen_gaze_x = self.gaze_x
        self.screen_gaze_y = self.gaze_y
        
    def check_zone_interaction(self):
        """Check if gaze is in an interaction zone and handle dwell time"""
        # Find which zone contains the current gaze point
        current_hit_zone = None
        
        if hasattr(self, 'zone_rects'):
            for i, rect in enumerate(self.zone_rects):
                if rect.contains(int(self.screen_gaze_x), int(self.screen_gaze_y)):
                    current_hit_zone = i
                    break
        
        # Reset dwell timer if gaze moved to a different zone
        if current_hit_zone != self.current_zone:
            self.current_zone = current_hit_zone
            self.dwell_start_time = time.time()
            self.zone_activated = False
        
        # Check if dwell threshold is met for the current zone
        if (self.current_zone is not None and 
            not self.zone_activated and 
            (time.time() - self.dwell_start_time) >= self.dwell_threshold):
            
            # Execute the action for this zone
            self.execute_zone_action(self.current_zone)
            self.zone_activated = True  # Prevent repeated activation
            
    def execute_zone_action(self, zone_index):
        """Execute the action associated with the given zone"""
        if 0 <= zone_index < len(self.zones):
            zone_name, action_callback, _ = self.zones[zone_index]
            if action_callback:
                action_callback()
                print(f"Activated zone: {zone_name}")
                
    def update_debug_info(self):
        debug_text = f"Gaze: ({int(self.screen_gaze_x)}, {int(self.screen_gaze_y)})\n"

        if self.stack.currentWidget() == self.main_control_widget:
            if self.current_zone is not None:
                zone_name = self.zones[self.current_zone][0]
                dwell_time = time.time() - self.dwell_start_time
                debug_text += f"Zone: {zone_name}\n"
                debug_text += f"Dwell: {dwell_time:.2f}s / {self.dwell_threshold:.2f}s"
            else:
                debug_text += "Zone: None"
        else:
            debug_text += "Zone: (không trong layout chính)"

        self.debug_label.setText(debug_text)

    def app_show_custom_notification(self, message, duration=2500, bg_color=QColor(0,150,200,210)):
        """Hiển thị một thông báo tùy chỉnh không chặn."""
        # Bước 1: Nếu có thông báo cũ đang hiển thị, yêu cầu nó đóng lại.
        # Không gán self.active_custom_notification_ref = None ngay ở đây.
        if self.active_custom_notification_ref and self.active_custom_notification_ref.isVisible():
            # print(f"DEBUG: Yêu cầu đóng thông báo cũ: {self.active_custom_notification_ref}")
            self.active_custom_notification_ref.close_smoothly()
            # Không nên truy cập self.active_custom_notification_ref nữa sau khi gọi close_smoothly
            # cho đến khi nó thực sự bị deleteLater hoặc bạn tạo cái mới.
            # Chúng ta sẽ tạo cái mới ngay sau đây, ghi đè lên tham chiếu cũ.

        # Bước 2: Tạo thông báo mới và lưu tham chiếu mới.
        new_notification = CustomNotificationWidget(
            message,
            parent=self,
            duration=duration,
            bg_color=bg_color
        )
        new_notification.show_centered_with_animation()
        self.active_custom_notification_ref = new_notification # Cập nhật tham chiếu
    # Zone action callbacks
    def call_caretaker(self):
        """Action for 'Gọi người chăm sóc' zone - Sử dụng thông báo tùy chỉnh."""
        print("App: Kích hoạt Gọi người chăm sóc!")

        if hasattr(self, 'app_reset_main_zone_after_action'):
             self.app_reset_main_zone_after_action(switch_widget=False)
        else:
             self.reset_zone_activation()

        self.app_show_custom_notification(
            "Đã gửi yêu cầu gọi người chăm sóc!",
            duration=3000,
            bg_color=QColor(46, 204, 113, 220) # Màu xanh lá cây
        )
        # (Tùy chọn) self.app_send_actual_caretaker_request_signal()


    def emergency(self):
        """Action for 'SOS' zone - Sử dụng thông báo tùy chỉnh không chặn (ĐÃ SỬA)."""
        print("App: Kích hoạt SOS!")

        if hasattr(self, 'app_reset_main_zone_after_action'):
             self.app_reset_main_zone_after_action(switch_widget=False)
        else:
             self.reset_zone_activation()

        self.app_show_custom_notification(
            "ĐÃ GỬI TÍN HIỆU KHẨN CẤP!",
            duration=3500,
            bg_color=QColor(200, 0, 0, 230) # Màu đỏ
        )    
    # <<<<< HÀM MỚI ĐỂ MỞ MUSIC MENU >>>>>
    def show_music_menu_action(self):
        print("Opening Music Menu...")
        # Cập nhật trạng thái nút play/pause trên music_menu_widget TRƯỚC KHI hiển thị
        current_media_state = self.media_player.state()
        is_playing_now = (current_media_state == QMediaPlayer.PlayingState)
        self.music_menu_widget.set_playing_indicator(is_playing_now)
        
        self.stack.setCurrentWidget(self.music_menu_widget)
        self.reset_zone_activation() 
        self.music_menu_widget.current_zone = None
        self.music_menu_widget.zone_activated = False
        self.music_menu_widget.dwell_start_time = None


    # <<<<< HÀM XỬ LÝ KHI CHỌN BÀI HÁT TỪ MUSIC MENU >>>>>
    def handle_music_selection(self, music_index):
        if not (0 <= music_index < len(self.music_files_paths)):
            QMessageBox.warning(self, "Lỗi nhạc", "Index bài hát không hợp lệ.")
            self.music_menu_widget.zone_activated = False # Reset zone trên widget con
            self.music_menu_widget.current_zone = None
            self.music_menu_widget.dwell_start_time = time.time()
            return

        file_path = self.music_files_paths[music_index]
        if file_path and os.path.exists(file_path):
            url = QUrl.fromLocalFile(file_path)
            content = QMediaContent(url)
            
            if self.media_player.state() != QMediaPlayer.StoppedState and self.current_playing_music_idx != music_index:
                self.media_player.stop()

            self.media_player.setMedia(content)
            self.media_player.play()
            self.current_playing_music_idx = music_index
            self.is_music_explicitly_paused = False # Khi chọn bài mới, không phải do người dùng pause
            
            track_name = self.music_menu_widget.music_track_names[music_index]
            QMessageBox.information(self, "Phát nhạc", f"Đang phát: {track_name}")
            print(f"Playing: {track_name} from {file_path}")
            self.music_menu_widget.set_playing_indicator(True) # Cập nhật UI
        else:
            QMessageBox.warning(self, "Lỗi nhạc", f"Không tìm thấy file: {file_path}")
            self.current_playing_music_idx = -1
            self.music_menu_widget.set_playing_indicator(False) # Cập nhật UI
        
        self.music_menu_widget.zone_activated = False
        self.music_menu_widget.current_zone = None
        self.music_menu_widget.dwell_start_time = time.time()


    # <<<<< HÀM XỬ LÝ TẠM DỪNG/TIẾP TỤC NHẠC >>>>>
    def handle_toggle_play_pause(self):
        current_state = self.media_player.state()
        is_playing_now = False

        if current_state == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.is_music_explicitly_paused = True # Người dùng chủ động pause
            print("Music paused by user.")
            is_playing_now = False
        elif current_state == QMediaPlayer.PausedState:
            # Chỉ play nếu trước đó là do người dùng pause, hoặc nếu có bài đang được chọn
            if self.is_music_explicitly_paused or self.current_playing_music_idx != -1:
                self.media_player.play()
                self.is_music_explicitly_paused = False
                print("Music resumed by user.")
                is_playing_now = True
            else: # Đang pause nhưng không phải do người dùng và không có bài nào -> không làm gì
                print("Music is paused but no track context to resume or not user initiated.")
                is_playing_now = False # Vẫn là paused
        elif current_state == QMediaPlayer.StoppedState and self.current_playing_music_idx != -1:
            # Nếu đã stop nhưng có bài hát trước đó, thử phát lại bài đó
            file_path = self.music_files_paths[self.current_playing_music_idx]
            if file_path and os.path.exists(file_path):
                url = QUrl.fromLocalFile(file_path)
                content = QMediaContent(url)
                self.media_player.setMedia(content)
                self.media_player.play()
                self.is_music_explicitly_paused = False
                print(f"Music re-played: {self.music_menu_widget.music_track_names[self.current_playing_music_idx]}")
                is_playing_now = True
            else:
                QMessageBox.warning(self, "Lỗi nhạc", "Không thể phát lại, file không tìm thấy.")
                is_playing_now = False
        else: # Chưa có nhạc hoặc trạng thái không xác định
            QMessageBox.information(self, "Thông báo", "Chưa có bài hát nào để phát/tạm dừng.")
            is_playing_now = False

        self.music_menu_widget.set_playing_indicator(is_playing_now) # Cập nhật UI
        self.music_menu_widget.zone_activated = False
        self.music_menu_widget.current_zone = None
        self.music_menu_widget.dwell_start_time = time.time()
   
    def read_book(self):
        """Action for 'Đọc sách' zone"""
        self.stack.setCurrentWidget(self.reading_menu)
        self.zone_activated = False
        self.current_zone = None
        self.dwell_start_time = time.time()
        
    
    def reset_zone_activation(self):
        """Reset zone activation to allow reactivation"""
        self.zone_activated = False
        self.current_zone = None
        self.dwell_start_time = 0
    
    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        self.cap.release()
        event.accept()

    def open_article(self, index):
        articles = [
            "Bài báo 1: Lorem ipsum dolor sit amet, consectetur adipiscing elit...",
            "Bài báo 2: Nulla facilisi. Sed sagittis leo nec bibendum.",
            "Bài báo 3: Curabitur cursus, eros eu luctus imperdiet...",
            "Bài báo 4: Vestibulum ante ipsum primis in faucibus..."
        ]
        content = articles[index] if index < len(articles) else "Bài viết đang được cập nhật..."
        self.reading_viewer.load_article(content)
        self.stack.setCurrentWidget(self.reading_viewer)
        if hasattr(self.reading_menu,'zone_activated'): # << Logic này đúng cho reading_menu
            self.reading_menu.zone_activated = False
            self.reading_menu.current_zone=None
            self.reading_menu.dwell_start_time=time.perf_counter()

    def go_home(self):
        self.stack.setCurrentWidget(self.main_control_widget)
        for widget in [self.reading_menu, self.reading_viewer, self.music_menu_widget]: # Thêm music_menu_widget vào list
            if hasattr(widget, 'update_gaze'):
                widget.update_gaze(-100, -100)
            if hasattr(widget, 'current_zone'): # Kiểm tra thuộc tính tồn tại
                widget.current_zone = None
            if hasattr(widget, 'zone_activated'):
                widget.zone_activated = False
            if hasattr(widget, 'dwell_start_time'):
                widget.dwell_start_time = None
        self.reset_zone_activation()

    def turn_page(self, direction):
        print(f"Chuyển trang: {direction}")
        # TODO: lật trang trong giao diện chọn bài

    def open_reading_menu(self):
        self.stack.setCurrentWidget(self.reading_menu)
        # Reset trạng thái của reading_viewer
        self.reading_viewer.gaze_x = -100
        self.reading_viewer.gaze_y = -100
        self.reading_viewer.zone_activated = False
        self.reading_viewer.dwell_start_time = time.time()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeTrackingControlApp()
    window.show()
    sys.exit(app.exec_())