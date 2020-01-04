# <빅콘테스트 2019 챔피언리그>

<div style="text-align: right; font-size:1.5em;"><b>-초코송이-</b> </div> 

## 0. 문제 정의
 - 엔씨소프트에서 제공하는 ‘리니지’ 고객 활동 데이터를 활용하여 향후 고객 이탈 방지를 위한 프로모션 수행 시 예상되는 잔존가치를 산정하는 예측 모형 개발
 - 참고: https://www.bigcontest.or.kr/points/content.php#ct04
 
## 1. 코드 실행에 필요한 패키지 및 라이브러리
- Anaconda3에 포함된 패키지
   - pandas 
   - numpy
   - sklearn
   - pickle 등
   - lightgbm (version 2.2.4)

## 2. 실행환경
- Anaconda3 기반의 python3 환경

## 3. 코드 실행 순서 및 방법
1. preprocess/preprocess.py
	- Train 데이터, Test 데이터 전처리 진행 코드
	- 실행 시, preprocess 폴더에 모델 학습을 위한 CSV 파일 생성
		- train\_preprocess\_1.csv 파일 (Survival Time을 예측하기 위한 모델을 학습하기 위한 Train 데이터)
		- train\_preprocess\_2.csv 파일 (Amount Spent를 예측하기 위한 모델을 학습하기 위한 Train 데이터)
	- 실행 시, preprocess 폴더에 Test 데이터를 예측하기 위한 모델의 input 데이터(CSV 파일) 생성
		- test1\_preprocess\_1.csv (Test1의 Survival Time을 예측하기 위한 Test1 전처리 데이터)
		- test1\_preprocess\_2.csv (Test1의 Amount Spent를 예측하기 위한 Test1 전처리 데이터)
		- test2_preprocess\_1.csv (Test2의 Survival Time을 예측하기 위한 Test2 전처리 데이터)
		- test2\_preprocess\_2.csv (Test2의 Amount Spent를 예측하기 위한 Test2 전처리 데이터)
2. model/create_model.py
	- 모델 틀 생성 및 preprocess 폴더 내 전처리 데이터를 기반으로 학습을 진행 뒤 학습된 모델 객체를 저장
		- train\_preprocess\_1.csv 파일로 Survival Time 예측을 위한 모델들을 학습
		- train\_preprocess\_2.csv 파일로 Amount Spent 예측을 위한 모델을 학습
	- model 폴더 내 Test 데이터 예측을 위한 모델 객체 생성
		- final\_model\_1.sav (Survival Time - 잔존/비잔존 유저를 예측하기 위한 분류 앙상블 모델)
		- final\_model\_2.sav (Survival Time - 비잔존 유저의 Survival Time을 예측하기 위한 회귀 앙상블 모델)
		- final\_model\_3.sav (Amount Spent - 유저의 Amount spent를 예측하기 위한 회귀 모델)
3. predict/predict.py 실행
	- model 폴더에 생성된 모델 객체와 preprocess 폴더에 있는 Test 전처리 데이터를 불러와서 survival time과 amount spent를 예측
	- 모델이 예측한 Test 데이터의 Survival Time과 Amount Spent의 예측 답안을 저장
		- Test1 : test1_predict.csv
		- Test2 : test2_predict.csv

## 4. 코드 정리
1. preprocess.py
	- 클래스 설명
  
	<table>
  		<tr>
  			<th colspan="2"><center>Class</center></th>
  		</tr>
  		<tr>
    		<th align='center'>이름</th>
    		<th align='center'>기능 설명</th>
  		</tr>
  		<tr>
  			<td align='center'><b>DataSet</b></td>
  			<td>raw 폴더에 있는 Train 데이터와 Test 데이터를 DataFrame 형식으로 불러옴<br/>
  			<b>get_train_data() 메소드 :</b><br/>
  			train 데이터에 있는 label, activity, payment, trade, pledge, combat 정보가 담긴 DataFrame들을 리턴해 줌<br/>
  			<b>get_test1_data() 메소드 :</b><br/>
  			test1 데이터에 있는 activity, payment, trade, pledge, combat 정보가 담긴 DataFrame들을 리턴해 줌
  			<b>get_test2_data() 메소드 :</b><br/>
  			test1 데이터에 있는 activity, payment, trade, pledge, combat 정보가 담긴 DataFrame들을 리턴해 줌
  			</td>
  		</tr>
	</table>
  
	- 함수 설명
  
	<table>
  		<tr>
  			<th colspan="2"><center>Function</center></th>
  		</tr>
  		<tr>
    		<th align='center'>이름</th>
    		<th align='center'>기능 설명</th>
  		</tr>
  		<tr>
  			<td align='center'><b>survival_time_preprocessing</b></td>
  			<td>survival time 모델의 input을 위한 데이터 전처리</td>
  		</tr>
  		<tr>
  			<td align='center'><b>amount_spent_preprocessing</b></td>
  			<td>amount spent 모델의 input을 위한 데이터 전처리</td>
  		</tr>
  		<tr>
  			<td align='center'><b>train_preprocessing</b></td>
  			<td>survival_time_preprocessing() 함수와 amount_spent_preprocessing() 함수를 이용하여 전처리를 진행한 뒤, label 데이터를 붙여서 survival time과 amount spent 학습을 위한 Train 데이터를 리턴해 줌</td>
  		</tr>
  		<tr>
  			<td align='center'><b>test_preprocessing</b></td>
  			<td>survival_time_preprocessing() 함수와 amount_spent_preprocessing() 함수를 이용하여 전처리를 진행한 뒤 survival time과 amount spent 예측을 위한 Test 데이터를 리턴해 줌</td>
  		</tr>
  		<tr>
  			<td align='center'><b>save_dataframe</b></td>
  			<td>주어진 DataFrame을 주어진 이름으로 저장해 줌</td>
  		</tr>
	</table>
  
	- 코드 진행
		- raw 데이터를 load
		- train/test 데이터 전처리
		- 전처리 된 데이터 저장

2. create\_model.py
	- 클래스 설명
  
	<table>
  		<tr>
  			<th colspan="2">Class</th>
  		</tr>
  		<tr>
    		<th align='center'>이름</th>
    		<th align='center'>기능 설명</th>
  		</tr>
  		<tr>
  			<td align='center'><b>DataSet</b></td>
  			<td>preprocess 폴더에 있는 전처리된 Train 데이터와 전처리된 Test 데이터를 DataFrame 형식으로 불러옴<br/>
  			<b>get_train_data() 메소드 :</b><br/>
  			survival time을 학습하기 위해 전처리된 train 데이터가 담긴 DataFrame과 amount spent를 학습하기 위해 전처리된 train 데이터가 담긴 DataFrame을 리턴해 줌<br/>
  			<b>get_test1_data() 메소드 :</b><br/>
  			survival time을 예측하기 위해 전처리된 test1 데이터가 담긴 DataFrame과 amount spent를 예측하기 위해 전처리된 test1 데이터가 담긴 DataFrame을 리턴해 줌<br/>
  			<b>get_test2_data() 메소드 :</b><br/>
  			survival time을 예측하기 위해 전처리된 test2 데이터가 담긴 DataFrame과 amount spent를 예측하기 위해 전처리된 test2 데이터가 담긴 DataFrame을 리턴해 줌
  			</td>
  		</tr>
  		<tr>
  			<td align='center'><b>_CustomBaseVoting</b></td>
  			<td>_CustomBaseVoting 클래스슬 상속받고, 앙상블 모델의 학습과 예측을 위한 기본적인 코드가 구현 되어있음, Base Class로 사용됨<br/>
  			<b>fit() 메소드 :</b><br/>
  			앙상블 모델의 복원/비복원 추출 학습을 위한 기능과 가중치 학습을 위한 기능이 구현되어 있음
        </td>
      </tr>
      <tr>
  			<td align='center'><b>CustomVotingClassifier</b></td>
  			<td>분류 앙상블 모델의 학습과 예측을 위한 클래스<br/>
  				<b>fit() 메소드 :</b><br/>
  				앙상블 모델의 복원/비복원 추출 학습을 위한 기능과 가중치 학습을 위한 메소드<br/>
  				<b>predict() 메소드 :</b><br/>
  				soft voting 방식을 이용하여 결과를 종합하여 predict한 값들을 리턴해 줌
  			</td>
      </tr>
  		<tr>
  			<td align='center'><b>CustomVotingRegressor</b></td>
  			<td>_CustomBaseVoting 클래스슬 상속받고, 회귀 앙상블 모델의 학습과 예측을 위한 클래스<br/>
  				<b>fit() 메소드 :</b><br/>
  				앙상블 모델의 복원/비복원 추출 학습을 위한 기능과 가중치 학습을 위한 메소드<br/>
  				<b>predict() 메소드 :</b><br/>
  				mean 방식을 이용하여 결과를 종합하여 predict한 값들을 리턴해 줌
  			</td>
  		</tr>
	</table>
  
	- 함수 설명
  
	<table>
  		<tr>
  			<th colspan="2"><center>Function</center></th>
  		</tr>
  		<tr>
    		<th align='center'>이름</th>
    		<th align='center'>기능 설명</th>
  		</tr>
  		<tr>
  			<td><b>make_lgb_models</b></td>
  			<td>주어진 LightGBM 모델 종류와 파라미터를 가진 LightGBM 모델을 주어진 개수만큼 만들어서 리스트 형태로 리턴해 줌 </td>
  		</tr>
  		<tr>
  			<td><b>custom_cost</b></td>
  			<td>model의 object function 정의<br/>
        grad, hess는 각각 score_function의 amount spent 예측값에 대한 1차, 2차 미분 값</td>
  		</tr>
  		<tr>
  			<td><b>custom_score</b></td>
  			<td>survival time의 예측값이 survival time의 실제값과 같다고 가정한 score function 정의<br/>
       또한 survival time의 예측값과 survival time의 실제값은 64일이 아니라는 가정함</td>
  		</tr>
  		<tr>
  			<td><b>train_model</b></td>
  			<td>survival time 예측을 위한 분류 앙상블 모델과 회귀 앙상블 모델, amount spent 예측을 위한 회귀 모델을 만듦<br/>주어진 survival time을 위한 전처리된 Train 데이터와 amount spent를 위한 전처리된 Train 데이터로 잔존/비잔존 예측 분류 앙상블 모델, 비잔존 유저의 survival time 예측 회귀 앙상블 모델, amount spent 예측 모델을 학습시켜서 리턴해 줌</td>
  		</tr>
  		<tr>
  			<td><b>save_model</b></td>
  			<td>주어진 모델 객체를 주어진 이름으로 저장해 줌</td>
  		</tr>
	</table>
  
	- 코드 진행
		- 전처리된 train 데이터들을 load
		- test 데이터 예측을 위한 모델들을 학습
		- 학습된 모델 객체들을 저장
3. predict.py
	- 클래스 설명

	<table>
  		<tr>
  			<th colspan="2"><center>Class</center></th>
  		</tr>
  		<tr>
    		<th align='center'>이름</th>
    		<th align='center'>기능 설명</th>
  		</tr>
  		<tr>
  			<td align='center'><b>DataSet</b></td>
  			<td>preprocess 폴더에 있는 전처리된 Test 데이터들을 DataFrame 형식으로 불러옴<br/>
  			<b>get_test1_data() 메소드 :</b><br/>
  			전처리된 test1 데이터들을 DataFrame 형태로 리턴해 줌<br/>
  			<b>get_test2_data() 메소드 :</b><br/>
  			전처리된 test2 데이터들을 DataFrame 형태로 리턴해 줌<br/>
  			</td>
  		</tr>
  		<tr>
  			<td align='center'><b>Models</b></td>
  			<td>model 폴더에 있는 학습된 모델 객체들을 불러옴<br/>
  			<b>predict() 메소드 :</b><br/>
  			주어진 survival time을 예측하기 위한 Test 전처리 데이터와 amount spent를 예측하기 위한 Test 전처리 데이터를 불러온 모델들을 이용하여 예측하고, survival time과 amount spent 예측값 정보들이 담긴 DataFrame을 리턴해 줌
  			</td>
  		</tr>
	</table>
  
	- 함수 설명
  
	<table>
  		<tr>
  			<th colspan="2"><center>Function</center></th>
  		</tr>
  		<tr>
    		<th align='center'>이름</th>
    		<th align='center'>기능 설명</th>
  		</tr>
  		<tr>
  			<td align='center'><b>survival_time_preprocessing</b></td>
  			<td>survival time 모델의 input을 위한 데이터 전처리</td>
  		</tr>
  		<tr>
  			<td align='center'><b>amount_spent_preprocessing</b></td>
  			<td>amount spent 모델의 input을 위한 데이터 전처리</td>
  		</tr>
  		<tr>
  			<td align='center'><b>train_preprocessing</b></td>
  			<td>survival_time_preprocessing() 함수와 amount_spent_preprocessing() 함수를 이용하여 전처리를 진행한 뒤, label 데이터를 붙여서 survival time과 amount spent 학습을 위한 Train 데이터를 리턴해 줌</td>
  		</tr>
  		<tr>
  			<td align='center'><b>test_preprocessing</b></td>
  			<td>survival_time_preprocessing() 함수와 amount_spent_preprocessing() 함수를 이용하여 전처리를 진행한 뒤 survival time과 amount spent 예측을 위한 Test 데이터를 리턴해 줌</td>
  		</tr>
  		<tr>
  			<td align='center'><b>save_dataframe</b></td>
  			<td>주어진 DataFrame을 주어진 이름으로 저장해 줌</td>
  		</tr>
	</table>
  
	- 코드 진행
		- raw 데이터를 load
		- train/test 데이터 전처리
		- 전처리 된 데이터 저장
