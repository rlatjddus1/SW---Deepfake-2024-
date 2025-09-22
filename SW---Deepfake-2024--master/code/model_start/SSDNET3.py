import os
import torch
import numpy as np
import pandas as pd
from models import SSDNet2D

# 모델 초기화
model = SSDNet2D()

state_dict = torch.load('./model_weight/weight_mixdrop10.pt', map_location=torch.device('cpu'))['MODEL_STATE']
model.load_state_dict(state_dict, strict=True) 

model.eval()  # 모델을 평가 모드로 설정 (옵션)

# CSV 파일 경로
existing_csv_path = './updated_test_origin_cqt.csv'

# CSV 파일 로드
df = pd.read_csv(existing_csv_path)

# 'human'이 1인 경우에 대해서만 예측 수행
results = []
for index, row in df.iterrows():
    if row['human'] > 0.56:
        file_path = row['path']

        # 예측할 데이터 로드
        data = np.load(file_path)
        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # 데이터를 PyTorch Tensor로 변환하고 (batch_size=1, channels=1, height, width) 차원 추가

        # 예측 수행
        with torch.no_grad():
            output = model(data)
        
        probabilities = torch.sigmoid(output)
        prob_real = probabilities[:, 1].item()  # Probability of "real"
        prob_fake = probabilities[:, 0].item()  # Probability of "fake"

        # 결과 저장
        results.append({
            'id': os.path.splitext(os.path.basename(file_path))[0],  # 파일 이름에서 확장자 제거하여 id로 사용
            'fake': prob_fake,
            'real': prob_real
        })
    else:
        file_path = row['path']

        # 예측할 데이터 로드
        data = np.load(file_path)
        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # 데이터를 PyTorch Tensor로 변환하고 (batch_size=1, channels=1, height, width) 차원 추가
        # 예측 수행
        with torch.no_grad():
            output = model(data) 

        probabilities = torch.sigmoid(output)
        prob_real = probabilities[:, 1].item() * row['human'] * 0  # Probability of "real"
        prob_fake = probabilities[:, 0].item() * row['human'] * 0  # Probability of "fake"
        
        results.append({
            'id': os.path.splitext(os.path.basename(file_path))[0],  # 파일 이름에서 확장자 제거하여 id로 사용
            'fake': prob_fake,
            'real': prob_real
        })

# 결과를 DataFrame으로 변환
df_results = pd.DataFrame(results)

# CSV 파일로 저장
csv_path = './result.csv'  # 결과 파일
df_results.to_csv(csv_path, index=False, encoding='utf-8')

print(f"Probabilities saved to {csv_path}")
