
# 使用卷積神經網路(CNN)解碼活化率(Firing Rate)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## 大綱

|標題|簡介|
|------------|-------------------------|
| 資料集介紹 | 介紹本範例使用之資料集 |
| 實驗介紹 | 介紹該資料及隻實驗流程 |
| 目標 | 本範例的主要想解決的任務   |
| 模型使用介紹 | 介紹解碼模型之細節  |

## 資料集介紹

主要利用活化率預測猴子手指的運動狀態，即位置、速度與加速度，三個變量。\
活化率數值高的地方恰與發生運動的區間(速度不為0的時間段)相符，\
故可利用活化率推得其運動狀態。

## 實驗介紹

以猴子I(Indy)為例，總計37個session。其資料的mat檔中各變量如下表所示：
| 名稱 | 用途   |
|------------|--------------------------|
| chan_naems | 各通道的所在腦區與編號 (**與spikes對照**)   |
| cursor_pos | 螢幕上猴子控制之光標位置 |
| finger_pos | 猴子手指位置             |
| spikes     | 棘波，共192筆資料         |
| t          | 當下時刻                 |
| target_pos | 螢幕上目標點位置         |
| wf         | 棘波波型(waveform)       |

當猴子移動手指時，置於其M1腦區的電極會接收到放電訊號(棘波, spikes)；\
而觸碰螢幕並移動光標的過程中，S1腦區的電極也會接收到放電訊號。\
上述兩個植入腦區的電極分別各有96個通道，總計192個。

## 目標

本範例僅使用M1腦區接收到之棘波並將各units相加(unsort)，\
最終會有96個通道的資料，加上角落4個無探針的資料，總計100個通道。\
接著以時間窗(time window)為64毫秒計算活化率；\
之後再從手指位置(mat檔中**finger_pos**資料)，推算速度與加速度兩個運動狀態。\
\
在本範例中會使用過去多個時間窗的數值，去預測當下時間窗的運動狀態。\
以擷取20個時間窗為例，假設預測的是當下時刻(<img src="http://chart.googleapis.com/chart?cht=tx&chl= ${t}$" style="border:none;">)的運動狀態，\
則會擷取<img src="http://chart.googleapis.com/chart?cht=tx&chl= ${(t-20}\sim{t)}$" style="border:none;">時刻的活化率進行預測，如下圖所示：
<img src=https://i.imgur.com/RanhWOs.png>
此處擷取的時間窗長度稱作"tap size (T)"。\
之後利用卷積神經網路擷取其特徵，藉此預測猴子手指移動之位置座標以及速度與加速度的數值。

## 模型使用介紹

本模型主要架構為卷積神經網路；編譯環境為：Win 10, python 3.8.5, pytorch 1.6.0。

### 步驟
### Step 1
下載資料並存放於"Data_I"的資料夾中。

### Step 2
下載程式並存放於前者之上級目錄中。

### Step 3
cd到所在目錄後，在cmd/terminal輸入指令切換至python環境，\
**PowerShell**
```ps
"VENV_PATH"/Scripts/Activate.ps1
```
**Cmd (Else)**
```ps
"VENV_PATH"/Scripts/activate.bat
```
接著使用python執行程式。
```ps
python main.py
```
各程式用途說明如下：
> [main.py](https://github.com/Abner0627/nc_lab_abner/blob/main/main.py)\
> 即主程式，有下列參數可以更改：\
> **1. Session**
> 更改讀取資料的session。
>  ```python
> single_session = bool(1)
> # 1為讀取單一session；0為讀取多個session的資料
> ```
> ```python
> if single_session:    
> # 讀取單一session
>     session = 0    
>     # 欲讀取session的編號，從0開始，共37個session
>     data_list = data_list[session:session+1]
> else:    
> # 讀取多個session
>     session_start = 0
>     # 從該session開始讀取
>     session_end = 0
>     # 至該session終止
>     data_list = data_list[session_start:session_end+1]
> ```
> **2. 訓練參數**
> ```python
> Epoch = 30
> lr = 1e-4    # Learning rate
> single_optim = optim.Adam(single_model.parameters(), lr=lr)    # Optimizer
> loss_MSE = nn.MSELoss()    # Loss function
> ```


> [model.py](https://github.com/Abner0627/nc_lab_abner/blob/main/model.py)\
> 模型的主要架構，使用卷積神經網路進行解碼。詳細架構圖如下：\
> <img src=https://i.imgur.com/c3IdQDs.png>


> [get_data.py](https://github.com/Abner0627/nc_lab_abner/blob/main/get_data.py)\
> 讀取猴子棘波的資料，並計算活化率；\
> 接著對資料做特徵縮放(feature scaling)，幫助梯度下降法(gradient decent)收斂。
> 預設為讀取所有session的資料，可在main.py中更改。


> [train.py](https://github.com/Abner0627/nc_lab_abner/blob/main/train.py)\
> 使用每個session前320秒的活化率與運動狀態訓練模型，其輸出為參數更新後的模型。


> [test](https://github.com/Abner0627/nc_lab_abner/blob/main/main.py#L96)\
> 使用每個session後320秒的活化率進行測試，\
> 最終輸出為預測結果的運動狀態，以及跟真實的運動狀態比較後的<img src="http://chart.googleapis.com/chart?cht=tx&chl= $R^2$" style="border:none;">。


> [comp.py](https://github.com/Abner0627/nc_lab_abner/blob/main/comp.py)\
> 紀錄預測結果的<img src="http://chart.googleapis.com/chart?cht=tx&chl= $R^2$" style="border:none;">值，並畫成直方圖表示。（第38項為平均結果）\
> <img src=https://i.imgur.com/cWZCfKS.png>
