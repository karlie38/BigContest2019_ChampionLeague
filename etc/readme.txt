<리니지 고객 활동 데이터를 활용하여 잔존 가치를 고려한 이탈 예측>
								-초코송이-
1. 코드 실행에 필요한 패키지 및 라이브러리
    - Anaconda3에 포함된 패키지
    - pandas
    - numpy
    - sklearn
    - pickle 등
    - lightgbm (version 2.2.4)
        version 2.2.3에서는 custom objective function이 포함된 lightgbm 모델을 저장하는 부분에서 오류발생
        오류없이 코드를 돌리기 위해서는 반드시 2.2.4 버전이 필요함

2. 실행환경
    - Anaconda3 기반의 python3 환경

3. 코드 실행 순서 및 방법
    1. 초코송이/preprocess/preprocess.py
        - 초코송이/raw 폴더에 있는 Train 데이터와 Test 데이터를 불러와서 전처리 하는 코드
        - 초코송이/preprocess 폴더에 모델 학습을 위한 CSV 파일 생성
            * train_preprocess_1.csv 파일 (Survival Time을 예측하기 위한 모델을 학습하기 위한 Train 데이터)
            * train_preprocess_2.csv 파일 (Amount Spent를 예측하기 위한 모델을 학습하기 위한 Train 데이터
        - 초코송이/preprocess 폴더에 Test 데이터를 예측하기 위한 모델의 input 데이터(CSV 파일) 생성
            * test1_preprocess_1.csv (Test1의 Survival Time을 예측하기 위한 Test1 전처리 데이터)
	* test1_preprocess_2.csv (Test1의 Amount Spent를 예측하기 위한 Test1 전처리 데이터)
	* test2_preprocess_1.csv (Test2의 Survival Time을 예측하기 위한 Test2 전처리 데이터)
	* test2_preprocess_2.csv (Test2의 Amount Spent를 예측하기 위한 Test2 전처리 데이터)
    2.초코송이/model/create_model.py
        - 모델을 생성하고, 초코송이/preprocess 폴더에 있는 전처리 데이터를 불러와서 학습을 진행한 뒤 학습된 모델 객체를 저장
            * train_preprocess_1.csv 파일로 Survival Time 예측을 위한 모델들을 학습
            * train_preprocess_2.csv 파일로 Amount Spent 예측을 위한 모델을 학습
        - 초코송이/model 폴더에 Test 데이터 예측을 위한 모델 객체 생성
            * final_model_1.sav (Survival Time - 잔존/비잔존 유저를 예측하기 위한 분류 앙상블 모델)
            * final_model_2.sav (Survival Time - 비잔존 유저의 Survival Time을 예측하기 위한 회귀 앙상블 모델)
            * final_model_3.sav (Amount Spent - 유저의 Amount spent를 예측하기 위한 회귀 모델)
    3. 초코송이/predict/predict.py 실행
        - 초코송이/model 폴더에 생성된 모델 객체를 불러와서 Test 데이터를 예측
        - 모델이 예측한 Test 데이터의 Survival Time과 Amount Spent의 예측 답안을 저장
            * Test1 : test1_predict.csv
            * Test2 : test2_predict.csv

4. 코드 정리
    - readme.md 파일에 자세히 정리