
# Eye Tracking Control Interface

Giao diện điều khiển bằng ánh nhìn (gaze) sử dụng camera và mô hình học máy để nhận diện vị trí ánh mắt người dùng và tương tác với giao diện GUI bằng PyQt5. Dự án này cung cấp hai chức năng chính:

- **Đọc báo bằng ánh mắt** với cuộn nội dung tự động.
- **Nghe nhạc** với giao diện chọn bài, chuyển trang, và điều khiển phát/tạm dừng chỉ bằng ánh nhìn.

---

## 🎯 Tính năng chính

### 📖 Reading Mode (Đọc báo)
- Giao diện lựa chọn bài viết (Reading Menu) với 2x2 bài trên mỗi trang, hỗ trợ chuyển trang bằng gaze.
- Tự động cuộn nội dung khi người dùng nhìn lên/xuống.
- Nhìn vào góc trái để quay lại menu.

### 🎵 Music Mode (Nghe nhạc)
- Hiển thị danh sách bài hát 3 bài/trang.
- Giao diện điều khiển nhạc: chọn bài, chuyển trang, phát/tạm dừng nhạc.
- Thanh hiển thị tiến trình phát nhạc.

### 👁️ Eye Tracking
- Sử dụng MediaPipe để lấy landmark khuôn mặt và tròng mắt.
- Trích xuất đặc trưng và sử dụng mô hình `GazeDualHeadMLP` để dự đoán tọa độ gaze.
- Dwell-based interaction: giữ ánh nhìn trong vùng tương tác để kích hoạt hành động.

---

## 🛠️ Công nghệ sử dụng

- **Ngôn ngữ:** Python
- **GUI:** PyQt5
- **Media xử lý:** OpenCV, MediaPipe
- **Mô hình học máy:** PyTorch
- **Gaze prediction:** MLP + Joblib Scaler
- **Multimedia playback:** QMediaPlayer

---

## 📁 Cấu trúc dự án

```
├── app.py                # File chính chạy ứng dụng
├── reading_menu.py       # Giao diện menu chọn bài báo
├── reading_viewer.py     # Giao diện đọc bài và cuộn tự động
├── music_menu.py         # Giao diện chọn nhạc và điều khiển nhạc
├── model/
│   ├── gaze_model.pth          # Trained PyTorch model
│   ├── scaler_input.joblib     # Joblib scaler for input features
│   └── scaler_target.joblib    # Joblib scaler for target gaze points
├── musics/                     # Thư mục chứa file nhạc .mp3
│   └── *.mp3
├── assets/                     # Icon cho giao diện (book, home, play, pause, ...)
│   └── *.
├── article/                     # Thư mục chứa file sách
│   └── *.txt
```

---

## 🚀 Hướng dẫn chạy

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

- Thư mục `musics/`: chứa các bài nhạc `.mp3`
- Thư mục `assets/`: chứa icon cho giao diện (`home_icon.png`, `music_icon.png`, `book_icon.png`, `play_icon.png`, ...)
- Thư mục `article/`: chứa các sách `.txt`

### 3. Chạy ứng dụng

```bash
python app.py
```

---

## 🧠 Mô hình học máy

- Mô hình `GazeDualHeadMLP` được huấn luyện để dự đoán tọa độ ánh nhìn từ các đặc trưng landmark trích xuất bởi MediaPipe.
- Các scaler `scaler_input.joblib` và `scaler_target.joblib` dùng để chuẩn hóa đầu vào/ra.

---

## 📌 Lưu ý

- Yêu cầu hệ thống có camera.
- Giao diện sử dụng toàn màn hình. Nhấn `ESC` để thoát.
- Đảm bảo đường dẫn các file nhạc và icon là hợp lệ.

---
