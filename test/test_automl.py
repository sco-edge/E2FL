# pip install autogluon
# example in 
# https://dacon.io/en/competitions/official/236075/codeshare/7764

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df.head()

# autogluon 학습을 위한 데이터 형태로 변환
train = TabularDataset(train_df.drop(['ID'], axis=1))
test = TabularDataset(test_df.drop(['ID'], axis=1))

# 이렇게 한 줄만 작성하면 내부에서 알아서 학습해줍니다.
predictor = TabularPredictor(label='전화해지여부', eval_metric='f1_macro',).fit(train)

# 각각의 모델의 훈련 성능을 평가할 수 있음
ld_board = predictor.leaderboard(train, silent=True)

# 예측하기
pred_y = predictor.predict(test)

# 제출 파일 생성
submit = pd.DataFrame()

submit['ID'] = test['ID']
submit['전화해지여부'] = pred_y

submit.to_csv('submit.csv', index=False)