# 쏘카 보험 사기 탐지 머신러닝 프로젝트
![image](https://user-images.githubusercontent.com/72847093/104838734-3476b900-5900-11eb-9428-96d19d7840d8.png)

## 개요 

### 프로젝트 주제
- 보험금을 목적으로 한 렌터카 사고 사기 건수가 증가
- 쏘카의 사고 데이터를 통해 Fraud 유저를 사전에 예측 및 예방 

### 프로젝트 진행순서
1. Settings
2. EDA 
3. Preprocessing 
4. Modulization 
5. Modeling 
6. Model Evaluation
 
### 시작에 앞서
- 본 프로젝트를 진행하기 위해서는 __Python 3__ 이상의 버젼과 다음의 설치가 필요합니다.
```
pip install numpy
pip install pandas
pip install seaborn
pip install matplotlib
pip install sklearn
pip install lightgbm 
pip install imblearn
pip install warnings
pip install sweetviz
pip install statsmodels
```

## 1. Settings

### 1-1. 환경설정
```python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
from statsmodels.stats.outliers_influence import variance_inflation_factor
# scaler 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC
# pipeline
from sklearn.pipeline import Pipeline 
# resampling
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
# model selection
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# scoring
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
# pca
from sklearn.decomposition import PCA
```

### 1-2. 데이터 불러오기
- 본 프로젝트는 쏘카로부터 데이터를 제공받아 진행된 프로젝트입니다. 
```python
# 1. 데이터 불러오기 
socar_df = pd.read_csv("insurance_fraud_detect_data.csv")
pd.set_option('display.max_columns', len(socar_df.columns))
socar = socar_df.copy()
```
## 2.EDA

### 2-1. SweetViz
```python
socar_tr = socar_df[socar_df["test_set"] == 0]
socar_test = socar_df[socar_df["test_set"] == 1]
socar_report = sv.compare([socar_tr, "Train"], [socar_test, "Test"], "fraud_YN")
socar_report.show_html('./socar_report.html')
```
<img src="https://user-images.githubusercontent.com/71831714/104716672-97831700-576b-11eb-80e5-867e81d60082.png" width='800'></img>


### 2-2. Seaborn

#### 1) 불균형한 데이터 분포
```python3
sns.countplot('fraud_YN', data=socar_df)
plt.title("Fraud Distributions \n", fontsize=14)
plt.show()
```
<img src="https://user-images.githubusercontent.com/71831714/105040354-16968900-5aa5-11eb-90bc-08845657fa94.png" width='400'></img>

#### 2) 컬럼별 분포도 확인
```python3
var = socar.columns.values

t0 = socar.loc[socar['fraud_YN']==0]
t1 = socar.loc[socar['fraud_YN']==1]

sns.set_style('whitegrid')
plt.figure()
fig,ax = plt.subplots(7,4,figsize=(16,28))

for i, feature in enumerate(var):
    plt.subplot(7,4,i+1)
    sns.kdeplot(t0[feature], bw=0.5, label = 'fraud_0')
    sns.kdeplot(t1[feature], bw=0.5, label = 'fraud_1')

    plt.xlabel(feature,fontsize=12)
    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which = 'major', labelsize=12)

plt.show()
```
<img src="https://user-images.githubusercontent.com/71831714/104717879-5b50b600-576d-11eb-9417-b8a123987454.png" width='600'></img>

#### 3) 상관관계 히트맵
```python3
mask = np.zeros_like(socar.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(socar.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
```
<img src="https://user-images.githubusercontent.com/71831714/104718021-9652e980-576d-11eb-868f-03c3c7843e5e.png" width='600'></img>

#### 4) 다중공선성
```python3
pd.DataFrame({"VIF Factor": [variance_inflation_factor(socar.values, idx) 
                             for idx in range(socar.shape[1])], "features": socar.columns})
```
<img src="https://user-images.githubusercontent.com/71831714/104718280-fea1cb00-576d-11eb-9ed3-d63b4d36eec4.png" width='200'></img>

#### 5) 변수 관찰
```python3
def make_graph(column):
    fig,ax = plt.subplots(2, 2, figsize=(20,12))

    t0 = socar[socar['fraud_YN']==0]
    t1 = socar[socar['fraud_YN']==1]

    plt.subplot(2,2,1)
    ax0 = sns.countplot(column, data=socar[socar['fraud_YN']==0])
    for p in ax0.patches:
        count = p.get_height()
        x = p.get_x() 
        y = p.get_y() + p.get_height()
        ax0.annotate(count, (x, y))
    plt.title("non-fraud {}".format(column))    

    plt.subplot(2,2,2)
    ax1 = sns.countplot(column, data=socar[socar['fraud_YN']==1])
    for p in ax1.patches:
        count = p.get_height()
        x = p.get_x() + 0.3
        y = p.get_y() + p.get_height()
        ax1.annotate(count, (x, y))
    plt.title("fraud {}".format(column))    

    plt.subplot(2,2,3)
    socar_df[socar_df['fraud_YN']==1][column].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title("fraud {}".format(column))    

    plt.subplot(2,2,4)
    sns.kdeplot(t0[column], bw=0.5, label = 'non-fraud')
    sns.kdeplot(t1[column], bw=0.5, label = 'fraud')
    plt.title("fraud vs non-fraud")   

    plt.show()

make_graph('accident_hour')
```
<img src="https://user-images.githubusercontent.com/71831714/105041042-dedc1100-5aa5-11eb-98f2-f05fed413e7a.png" width='600'></img>

## 3. Modulization
 - 코드의 간결성을 위해 모듈화 진행
 
<img src="https://user-images.githubusercontent.com/71831714/105041949-01baf500-5aa7-11eb-98de-67aeb1a13db2.png" width='400'></img>
<img src="https://user-images.githubusercontent.com/71831714/105041951-02ec2200-5aa7-11eb-9732-d91191eb0f26.png" width='400'></img>

## 4. Preprocessing 

### 4-1. 결측치처리 & 이상치 제거 
- 평균값/중앙값/최빈값/KNN imputer 를 활용한 결측치 보간 진행 

### 4-2. 원핫인코딩(OneHotEncoding)
- 적용 안 함 
- 모든 카테고리 변수에 적용

### 4-3. 스케일링(Scaling)
- StandardScaler
- MinMaxScaler 
- RobustScaler
- Log Scailing 

### 4-4. 샘플링(Sampling)
Imbalanced Data 처리를 위한 다양한 샘플링 기법 시도 

<img src="https://user-images.githubusercontent.com/71831714/105041213-0d59ec00-5aa6-11eb-93e4-5d3eaedb94c2.png" width='500'></img>

- RandomOverSampler
- SMOTE 
- ADASYN
- RandomUnderSampler
- SMOTETomek
- SMOTEENN

### 4-5. 주성분 분석(PCA)
- 차원 축소 기법을 통한 데이터 노이즈 제거 

## 5. Modeling  
### 분류기(Classifier)
- LogisticRegression
- DecisionTree
- RandomForest
- LGBM
- LinearSVC 

### 하이퍼 파라미터 튜닝(Hyper Parameter Tuning)
 - 최적의 하이퍼 파라미터 값을 찾기 위해 교차 검증 사용 
 
<img src="https://user-images.githubusercontent.com/71831714/105041678-9ffa8b00-5aa6-11eb-9bd0-59cef68e42c6.png" width='500'></img>


## 6. Model Evaluation 
### 모델 성능 평가
- 재현률(Recall)과 정밀도(Accuracy)를 기준으로 성능 평가 진행 

<img src="https://user-images.githubusercontent.com/71831714/105040231-ed75f880-5aa4-11eb-83f9-d94772f72028.png" width='400'></img>
<img src="https://user-images.githubusercontent.com/71831714/105040236-ef3fbc00-5aa4-11eb-9c02-7715142bcffb.png" width='400'></img>
 
### 최고의 모델(Best Model)

- 모델 1

<img src="https://user-images.githubusercontent.com/71831714/105040398-24e4a500-5aa5-11eb-95a7-9a0a2e107fb4.png" width='400'></img>
<img src="https://user-images.githubusercontent.com/71831714/105040406-2615d200-5aa5-11eb-9bbd-9c9e80854f8c.png" width='400'></img>

    acc_type1, b2b, repair_cost, car_part1, car_part2, repair_cnt, insurance_site_aid_YN, police_site_aid_YN 컬럼 제거
    원핫인코딩
    StandardScaling
    RandomUnderSampling
    주성분 분석으로 데이터를 4차원으로 축소
    DecisionTree max_depth를 4로 지정
    
 1) validation set과 test set 모두에서 비슷한 성적을 보여줌
 2) Fraud 데이터 7건 중 5건을 잡아내 높은 recall 기록
 3) accuracy가 낮다는 한계점 존재

## Conclusion
- Random Under Sampling이 다른 샘플링 기법들보다 좋은 성능을 보여준 이유에 대해 추가 학습 예정
- 차원을 축소함으로서 속도 뿐만 아니라 성능이 크게 향상됨
- 좀 더 다양한 Hyper Parameter에 대해 학습할 시간에 부족한 점이 아쉬움

## 함께한 분석가 :thumbsup:

- 김성준
  - EDA, Preprocessing, Modulization, Modeling, Readme
  - GitHub: https://github.com/alltimeno1
    
- 김미정 
  - EDA, Preprocessing,Modeling, ppt
  - GitHub: https://github.com/LeilaYK
  
- 이정려
  - EDA, Preprocessing,Modeling, ppt
  - GitHub: https://github.com/jungryo
  
- 전예나
  - EDA, Preprocessing,Modeling, ppt, presentation
  - GitHub: https://github.com/Yenabeam
