import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, 0, :])
        return out
    
class Stock:
    def __init__(self, df):
        super(Stock, self).__init__()
        self.df=df.copy()
        self.scaler_all = MinMaxScaler(feature_range=(0,1))
        self.scaler_target = MinMaxScaler(feature_range=(0,1))
        self.data=df.values
        self.predictions = []  # 예측값을 저장할 리스트
        self.actuals = []  # 실제값을 저장할 리스트
        self.train_losses = []
        self.val_losses = []
        

    def preprocessing(self):
        # null값 평균값 대체
        self.df.loc[self.df['Volume'] == 0, 'Volume'] = np.mean(self.df[self.df['Datetime'].str.contains('09:00:00\+09:00')]['Volume'].values)
        # index를 날짜로 설정
        self.df["Datetime"] = pd.to_datetime(self.df["Datetime"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
        self.df.set_index('Datetime', inplace=True)        
 

    def add_change(self, columns):
        for col in columns:
            self.df[f'{col}_chg']=self.df[col].pct_change()
        self.df.dropna(inplace=True)


    
    def add_col(self):
        self.df['change']=0
        self.df['target']=0
        for i in range(len(self.df)-1):
            self.df.iloc[i+1,5]=self.df.iloc[i+1,3]-self.df.iloc[i,3]
        self.df.loc[self.df['change']==0, 'target']=0
        self.df.loc[self.df['change']<0, 'target']=-1
        self.df.loc[self.df['change']>0, 'target']=1


    def scale_col(self, selected_feature):
        self.selected_feature=selected_feature
        data=self.df[selected_feature].values
        self.data = self.scaler_all.fit_transform(data)
        self.scaler_target.fit_transform(data[:,0].reshape(1,-1))
        self.scaler_target.min_, self.scaler_target.scale_ = self.scaler_all.min_[0], self.scaler_all.scale_[0]


    def create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data)-seq_length):
            x = data[i:(i+seq_length), ]
            y = data[i+seq_length, 0]  # 예측하려는 값을 0에 배치
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    def data_loader(self, seq_len, type='train'):
        self.seq_len=seq_len
        train_size = int(len(self.data) * 0.7)
        val_size = int(len(self.data) * 0.2)
        test_size = len(self.data) - train_size - val_size

        if type=='train':
            X, y = self.create_sequences(self.data[:train_size], seq_len)
        elif type=='valid':
            X, y = self.create_sequences(self.data[train_size:train_size+val_size], seq_len)
        elif type=='test':
            X, y = self.create_sequences(self.data[train_size+val_size:], seq_len)
        else:
            X, y = self.create_sequences(self.data, seq_len)
                
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        data = TensorDataset(X, y)
        data_loader = DataLoader(dataset=data, batch_size=16, shuffle=False)

        return data_loader
    
    def create_model(self, input_size=6, hidden_size = 256, output_size = 1):
        # 모델 생성
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size)
    
    def train(self, train_loader, val_loader, test_loader, type, num_epoch = 100, min_delta=0.00001):
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        # 손실 함수 설정
        criterion = nn.MSELoss()

        # 학습 파라미터 설정
        self.num_epochs = num_epoch
        self.train_losses = []
        self.val_losses = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        min_delta = 0.00001  # 개선으로 간주되기 위한 최소 변화량
        best_loss = np.inf  # 가장 낮은 검증 손실을 추적dd
        last_val_loss = np.inf
    
        # 모델 학습
        if type == 'train':
            for epoch in range(self.num_epochs):
                self.model.train()  # 학습 모드로 설정
                train_loss = 0.0
                
                for inputs, labels in train_loader:
                    # 입력 데이터와 레이블을 GPU로 이동
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # 기울기 초기화
                    optimizer.zero_grad()
                    
                    # 순방향 전파
                    outputs = self.model(inputs)
                    
                    # 손실 계산
                    loss = criterion(outputs.squeeze(), labels)
                    
                    # 역전파 및 가중치 업데이트
                    loss.backward()
                    optimizer.step()
                    
                    # 손실 누적
                    train_loss += loss.item() * inputs.size(0)
            
                train_loss /= len(train_loader.dataset)
                self.train_losses.append(train_loss)

                self.model.eval()  # 모델을 평가 모드로 설정
                val_loss = 0.0

                with torch.no_grad():
                    self.model.eval()
                    for inputs, labels in val_loader:
                        outputs = self.model(inputs)
                        loss = criterion(outputs.squeeze(), labels)
                        val_loss += loss.item() * inputs.size(0)  # 누적 손실 계산
                    # 에포크별 평균 검증 손실 계산
                    val_loss /= len(val_loader.dataset)
                    self.val_losses.append(val_loss)

                print(f'Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
                patience = 0
                if val_loss > best_loss + min_delta:
                    patience += 1
                    if patience == 25:
                        print("Early stopping initiated.")
                        print(f"Best Validation Loss: {best_loss:.5f}")
                        break
                else:
                    best_loss = val_loss
                    patience = 0
                    

                last_val_loss = val_loss  # 마지막 검증 손실 업데이트


            # 테스트 데이터셋으로 평가
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = self.model(inputs)
                    test_loss += criterion(outputs, targets.unsqueeze(1)).item()

            print(f'Test Loss (MSE): {test_loss/len(test_loader):.4f}')
        
        if type == 'test':
            self.predictions=[]
            self.actuals=[]
            self.model.eval()  # 모델을 평가 모드로 설정
            test_losses = []  # 테스트 손실을 저장할 리스트

            with torch.no_grad():  # 기울기 계산을 비활성화
                for seqs, labels in test_loader:

                    outputs = self.model(seqs)

                    # 손실 계산
                    loss = criterion(outputs, labels)
                    test_losses.append(loss.item())

                    # 예측값과 실제값 저장
                    self.predictions.extend(outputs.view(-1).detach().numpy())
                    self.actuals.extend(labels.view(-1).detach().numpy())

            # 평균 테스트 손실 계산 및 출력
            average_test_loss = sum(test_losses) / len(test_losses)
            print(f'Average Test Loss: {average_test_loss}')
        
  

    def pred_value(self, type):
        if (type=='chg')|(type=='t'):
            train_size = int(len(self.data) * 0.7)
            val_size = int(len(self.data) * 0.2)
            if type=='t':
                yest=self.df.iloc[self.seq_len:,3].values.reshape(-1,1)
            else:
                yest=self.df.iloc[train_size+val_size+self.seq_len:,3].values.reshape(-1,1)
            self.predictions_inverse = np.round(self.scaler_target.inverse_transform(np.array(self.predictions).reshape(-1,1))*0.01+yest, -2)
            self.actuals_inverse = np.round(self.scaler_target.inverse_transform(np.array(self.actuals).reshape(-1,1)), -2)
        else:
            self.predictions_inverse = np.round(self.scaler_target.inverse_transform(np.array(self.predictions).reshape(-1,1)), -2)
            self.actuals_inverse = np.round(self.scaler_target.inverse_transform(np.array(self.actuals).reshape(-1,1)), -2)
        return self.predictions_inverse, self.actuals_inverse
    
