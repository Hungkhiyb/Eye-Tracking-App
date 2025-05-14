import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QLabel, QPushButton, QComboBox, QSlider, QGroupBox, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QFont, QPalette, QColor, QImage, QPixmap, QPainter, QPen, QLinearGradient
import cv2
import numpy as np
import torch
import mediapipe as mp
from train import GazeDualHeadMLP
from joblib import load
import pyautogui
import math
import time
import os # Thêm os để làm việc với đường dẫn file

# Import pygame
import pygame
class EyeTrackingControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracking Control Interface")

        # Khởi tạo pygame mixer (chỉ một lần)
        try:
            pygame.mixer.init()
            self.pygame_mixer_initialized = True
            print("Pygame mixer initialized successfully.")
        except pygame.error as e:
            self.pygame_mixer_initialized = False
            print(f"Error initializing pygame mixer: {e}")
            QMessageBox.warning(self, "Lỗi Mixer", 
                                "Không thể khởi tạo trình phát nhạc (pygame.mixer).\n"
                                "Chức năng phát nhạc có thể không hoạt động.")

        # Đường dẫn đến file nhạc của bạn
        # THAY ĐỔI "path/to/your/music.mp3" THÀNH ĐƯỜNG DẪN THỰC TẾ
        self.music_file_path = "Let Her Go.mp3"
        
        self.init_eye_tracking()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            sys.exit()

        # Setup main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

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
        # self.zones = [
        #     ("Gọi người chăm sóc", self.call_caretaker, QColor("#ef5350")),  # đỏ nhạt
        #     ("Bật nhạc", self.play_music, QColor("#66bb6a")),                # xanh lá
        #     ("Đọc sách", self.read_book, QColor("#42a5f5")),                 # xanh dương
        #     ("SOS", self.emergency, QColor("#d32f2f")),                      # đỏ đậm
        #     ("", None, QColor("#424242")),                                   # xám
        #     ("", None, QColor("#424242")),                                   # xám
        # ]
        self.zones = [
            ("Gọi người chăm sóc", self.call_caretaker, QColor("#ef5350")),
            ("Bật nhạc", self.play_music, QColor("#66bb6a")),
            ("Đọc sách", self.read_book, QColor("#42a5f5")),
            ("SOS", self.emergency, QColor("#d32f2f")),
            ("Dừng nhạc", self.stop_music, QColor("#ffca28")), # Thêm nút dừng nhạc (màu vàng)
            ("", None, QColor("#424242")),
        ]

        
        # Zone rectangles will be calculated during paintEvent based on window size
        
    def paintEvent(self, event):
        painter = QPainter(self)

        # Get window dimensions
        window_width = self.width()
        window_height = self.height()

        # Zone size
        # zone_width = (window_width - 40) // 3
        # zone_height = (window_height - 40) // 2
        num_zones = len(self.zones)
        cols = 3 # Số cột mong muốn
        rows = (num_zones + cols - 1) // cols # Tính số hàng cần thiết

        # Tính toán kích thước vùng dựa trên số hàng và cột
        zone_width = (window_width - (cols + 1) * 10) // cols
        zone_height = (window_height - (rows + 1) * 10) // rows
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
                self.check_zone_interaction()
                
                # Update debug info
                self.update_debug_info()
                
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
        """Update the debug information display"""
        debug_text = f"Gaze: ({int(self.screen_gaze_x)}, {int(self.screen_gaze_y)})\n"
        
        if self.current_zone is not None:
            zone_name = self.zones[self.current_zone][0]
            dwell_time = time.time() - self.dwell_start_time
            debug_text += f"Zone: {zone_name}\n"
            debug_text += f"Dwell: {dwell_time:.2f}s / {self.dwell_threshold:.2f}s"
        else:
            debug_text += "Zone: None"
            
        self.debug_label.setText(debug_text)
        
    # Zone action callbacks
    # def call_caretaker(self):
    #     """Action for 'Gọi người chăm sóc' zone"""
    #     QMessageBox.information(self, "Thông báo", "Đã gọi người chăm sóc!")
    #     # Reset zone activation after a short delay
    #     QTimer.singleShot(500, self.reset_zone_activation)
        
    # def play_music(self):
    #     """Action for 'Bật nhạc' zone"""
    #     QMessageBox.information(self, "Thông báo", "Đang bật nhạc!")
    #     # Reset zone activation after a short delay
    #     QTimer.singleShot(500, self.reset_zone_activation)
        
    # def read_book(self):
    #     """Action for 'Đọc sách' zone"""
    #     QMessageBox.information(self, "Thông báo", "Mở ứng dụng đọc sách!")
    #     # Reset zone activation after a short delay
    #     QTimer.singleShot(500, self.reset_zone_activation)
        
    # def emergency(self):
    #     """Action for 'SOS' zone"""
    #     QMessageBox.critical(self, "KHẨN CẤP", "ĐÃ GỬI TÍN HIỆU KHẨN CẤP!")
    #     # Reset zone activation after a short delay
    #     QTimer.singleShot(500, self.reset_zone_activation)
    
    def call_caretaker(self):
        """Action for 'Gọi người chăm sóc' zone"""
        trigger_x = int(self.screen_gaze_x) # Đảm bảo self.screen_gaze_x có giá trị số
        trigger_y = int(self.screen_gaze_y) # Đảm bảo self.screen_gaze_y có giá trị số
        
        # In ra để kiểm tra giá trị
        print(f"DEBUG: call_caretaker triggered. Coords: ({trigger_x}, {trigger_y})")

        # Tạo chuỗi message
        message_content = f"Đã gọi người chăm sóc!\nKích hoạt tại tọa độ màn hình: ({trigger_x}, {trigger_y})"
        
        # In ra để kiểm tra nội dung message
        print(f"DEBUG: Message content is: '{message_content}'")

        # Kiểm tra xem message_content có rỗng không
        if not message_content:
            print("ERROR: Message content is EMPTY!")
            message_content = "LỖI: Nội dung thông báo rỗng!" # Thông báo mặc định nếu rỗng

        QMessageBox.information(self, "Thông báo", message_content)
        
        print(f"Hành động 'Gọi người chăm sóc' được kích hoạt tại ({trigger_x}, {trigger_y}).")
        QTimer.singleShot(500, self.reset_zone_activation)
        
    def play_music(self):
        """Hành động cho vùng 'Bật nhạc'"""
        if not self.pygame_mixer_initialized:
            QMessageBox.warning(self, "Lỗi nhạc", "Trình phát nhạc chưa được khởi tạo.")
            QTimer.singleShot(500, self.reset_zone_activation)
            return

        trigger_x = int(self.screen_gaze_x)
        trigger_y = int(self.screen_gaze_y)
        
        try:
            if not os.path.exists(self.music_file_path):
                QMessageBox.critical(self, "Lỗi file nhạc", f"Không tìm thấy file nhạc:\n{self.music_file_path}")
                QTimer.singleShot(500, self.reset_zone_activation)
                return

            if pygame.mixer.music.get_busy():
                # Nếu nhạc đang phát, có thể chọn dừng hoặc thông báo
                pygame.mixer.music.stop() # Dừng nhạc cũ để phát lại từ đầu
                # pygame.mixer.music.unload() # Cân nhắc unload nếu muốn giải phóng bộ nhớ
                print("Music was playing, stopped it to replay.")
            
            pygame.mixer.music.load(self.music_file_path)
            pygame.mixer.music.play() # Mặc định play 1 lần, dùng play(-1) để lặp vô hạn
            
            song_name = os.path.basename(self.music_file_path)
            message = f"Đang phát: {song_name}\nKích hoạt tại: ({trigger_x}, {trigger_y})"
            QMessageBox.information(self, "Bật nhạc", message)
            print(f"Playing music: {self.music_file_path}")

        except pygame.error as e:
            QMessageBox.critical(self, "Lỗi phát nhạc", f"Lỗi Pygame Mixer: {e}")
            print(f"Pygame mixer error: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi không xác định", f"Lỗi khi bật nhạc: {e}")
            print(f"Unexpected error in play_music: {e}")
            
        QTimer.singleShot(1000, self.reset_zone_activation) # Tăng thời gian reset


    def stop_music(self):
        """Hành động cho vùng 'Dừng nhạc'"""
        if not self.pygame_mixer_initialized:
            QMessageBox.warning(self, "Lỗi nhạc", "Trình phát nhạc chưa được khởi tạo.")
            QTimer.singleShot(500, self.reset_zone_activation)
            return

        trigger_x = int(self.screen_gaze_x)
        trigger_y = int(self.screen_gaze_y)

        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            # pygame.mixer.music.unload() # Cân nhắc unload nếu muốn giải phóng hoàn toàn
            message = f"Đã dừng nhạc.\nKích hoạt tại: ({trigger_x}, {trigger_y})"
            QMessageBox.information(self, "Dừng nhạc", message)
            print("Music stopped.")
        else:
            message = f"Không có nhạc nào đang phát.\nTại: ({trigger_x}, {trigger_y})"
            QMessageBox.information(self, "Dừng nhạc", message)
            print("No music was playing.")
        
        QTimer.singleShot(1000, self.reset_zone_activation)
        
    def read_book(self):
        """Hành động cho vùng 'Đọc sách'"""
        trigger_x = int(self.screen_gaze_x)
        trigger_y = int(self.screen_gaze_y)

        message = f"Mở ứng dụng đọc sách!\nKích hoạt tại tọa độ màn hình: ({trigger_x}, {trigger_y})"
        QMessageBox.information(self, "Thông báo", message)
        
        print(f"Hành động 'Đọc sách' được kích hoạt tại ({trigger_x}, {trigger_y}).")
        # Ví dụ: mở một cuốn sách cụ thể hoặc một trang dựa trên tọa độ
        # open_reading_app_to_page(book_id, page_based_on_coords)

        QTimer.singleShot(500, self.reset_zone_activation)
        
    def emergency(self):
        """Hành động cho vùng 'SOS'"""
        trigger_x = int(self.screen_gaze_x)
        trigger_y = int(self.screen_gaze_y)

        message = f"ĐÃ GỬI TÍN HIỆU KHẨN CẤP!\nKích hoạt tại tọa độ màn hình: ({trigger_x}, {trigger_y})"
        QMessageBox.critical(self, "KHẨN CẤP", message)
        
        print(f"Hành động 'SOS' được kích hoạt tại ({trigger_x}, {trigger_y}).")
        # Gửi tín hiệu SOS, có thể kèm theo thông tin vị trí nếu có
        # (lưu ý: screen_gaze_x/y là tọa độ trên màn hình, không phải vị trí địa lý)
        # send_sos_signal(additional_info={"trigger_coords": (trigger_x, trigger_y)})

        QTimer.singleShot(500, self.reset_zone_activation)
    
    def reset_zone_activation(self):
        """Reset zone activation to allow reactivation"""
        self.zone_activated = False
        self.current_zone = None
        self.dwell_start_time = 0
    
    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeTrackingControlApp()
    window.show()
    sys.exit(app.exec_())