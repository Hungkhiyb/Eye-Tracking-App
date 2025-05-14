import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import MultiheadAttention


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# ==== MÔ HÌNH ====
import torch.nn as nn
import torch.nn.functional as F

class GazeDualHeadMLP(nn.Module):
    def __init__(self, input_size=976, shared_sizes=[1024, 1024, 512, 512, 256], head_sizes=[256, 128, 64], num_heads=8):
        super().__init__()
        
        # Phần shared với residual connections
        self.shared = nn.ModuleList()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, shared_sizes[0]),
            nn.BatchNorm1d(shared_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4)
        )
        
        # Các residual blocks
        for i in range(len(shared_sizes)-1):
            self.shared.append(
                ResidualBlock(shared_sizes[i], shared_sizes[i+1], dropout=0.3 if i < 2 else 0.2)
            )
        
        # Multi-head attention layer
        self.attention = MultiheadAttention(embed_dim=shared_sizes[-1], num_heads=num_heads, dropout=0.2, batch_first=True)
        self.attention_norm = nn.LayerNorm(shared_sizes[-1])
        
        # Heads phức tạp hơn với skip connections
        self.head_x = AdvancedHead(shared_sizes[-1], head_sizes)
        self.head_y = AdvancedHead(shared_sizes[-1], head_sizes)
        
        # Khởi tạo trọng số
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Residual blocks
        for layer in self.shared:
            x = layer(x)
        
        # Attention mechanism
        attn_input = x.unsqueeze(1)  # Thêm dimension sequence
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        x = x + 0.5 * attn_output.squeeze(1)
        x = self.attention_norm(x)
        
        # Dual heads
        out_x = self.head_x(x)
        out_y = self.head_y(x)
        
        return torch.cat([out_x, out_y], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        # Skip connection nếu kích thước thay đổi
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = self.leakyrelu(out)
        return out


class AdvancedHead(nn.Module):
    def __init__(self, input_size, head_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Tạo các layer với skip connections
        prev_size = input_size
        for size in head_sizes:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(prev_size, size),
                    nn.BatchNorm1d(size),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2)
                )
            )
            prev_size = size
        
        # Output layer
        self.output = nn.Linear(head_sizes[-1], 1)
        
        # Skip connection từ input đến gần output
        if input_size != head_sizes[-1]:
            self.skip = nn.Sequential(
                nn.Linear(input_size, head_sizes[-1]),
                nn.BatchNorm1d(head_sizes[-1])
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        out = x
        for layer in self.layers:
            out = layer(out)
        out = out + residual
        out = self.output(out)
        return out


# ==== LOSS ====
def weighted_mse_loss(pred, target, weight_y=3.0):
    loss_x = F.mse_loss(pred[:, 0], target[:, 0])
    loss_y = F.mse_loss(pred[:, 1], target[:, 1])
    return loss_x + weight_y * loss_y

# def hybrid_loss(pred, target, weight_y=3.0):
#     # Kết hợp MSE và Angular loss
#     mse_loss = F.mse_loss(pred, target)
    
#     # Tính góc giữa vector dự đoán và target
#     pred_vec = pred - torch.mean(pred, dim=0)
#     target_vec = target - torch.mean(target, dim=0)
#     cos_sim = F.cosine_similarity(pred_vec, target_vec)
#     angular_loss = 1 - cos_sim.mean()
    
#     return mse_loss + 0.5 * angular_loss

# ==== HUẤN LUYỆN ====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    data = pd.read_csv("gaze_dataset_with_features.csv")

    # ==== CHUẨN HÓA ====
    scaler_input = MinMaxScaler()
    scaler_target = MinMaxScaler()

    landmark_columns = [col for col in data.columns if col.startswith(('x_', 'y_'))]
    special_features = [
        "eye_vert_dist", "iris_ratio",
        "norm_x_left", "norm_y_left", "norm_x_right", "norm_y_right",
        "ear_left", "ear_right",
        "lid_ratio_left", "lid_ratio_right",
        "velocity", "acceleration",
        "illum_mean", "illum_std",
        "pitch", "yaw", "roll",
        "trans_x", "trans_y", "trans_z"
    ]
    target_columns = ['target_x', 'target_y']

    X = scaler_input.fit_transform(data[landmark_columns + special_features].values)
    y = scaler_target.fit_transform(data[target_columns].values)

    dump(scaler_input, 'scaler_input.joblib')
    dump(scaler_target, 'scaler_target.joblib')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # ==== MÔ HÌNH ====
    model = GazeDualHeadMLP(input_size=X_train_tensor.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=5e-5,
        steps_per_epoch=len(X_train_tensor) // 64 + 1,
        epochs=300,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # ==== EARLY STOPPING ====
    best_val_loss = float('inf')
    patience = 50
    wait = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    epochs = 500
    batch_size = 128

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        total_loss = 0

        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = weighted_mse_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # ==== VALIDATION ====
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = weighted_mse_loss(val_outputs, y_test_tensor)

        train_losses.append(total_loss)
        val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ==== EARLY STOPPING ====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # ==== LƯU MÔ HÌNH ====
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), 'gaze_model.pth')

    # ==== ĐÁNH GIÁ CUỐI ====
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions = scaler_target.inverse_transform(predictions.cpu().numpy())
        y_test_original = scaler_target.inverse_transform(y_test_tensor.cpu().numpy())

        distances = np.sqrt(np.sum((predictions - y_test_original)**2, axis=1))

        print('\n=== Final Evaluation ===')
        print(f'Mean Distance: {np.mean(distances):.2f} pixels')
        print(f'Median Distance: {np.median(distances):.2f} pixels')
        print(f'Accuracy <50px: {np.mean(distances < 50) * 100:.2f}%')


    # ==== VẼ BIỂU ĐỒ ====
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(distances, bins=30, color='skyblue', edgecolor='black')
    plt.title('Prediction Error (pixels)')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
