import torch
import torch.nn as nn

# --- 하이퍼파라미터 및 설정 ---
NUM_CLASSES = 5
NUM_FEATURES = 335

# 1D-CNN + LSTM 모델입니다!
# --- 모델 아키텍처 정의 ---
class SignLanguageModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SignLanguageModel, self).__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 입력 데이터의 차원이 2차원일 경우 배치 차원을 추가
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        final_output = lstm_out[:, -1, :] 
        logits = self.classifier(final_output)
        
        return logits