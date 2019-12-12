import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import sys
sys.path.append("../model")
from create_model import _CustomBaseVoting, CustomVotingClassifier, CustomVotingRegressor, \
                         custom_cost, custom_score
#from sklearn.externals import joblib

class DataSet:
    def __init__(self):
        self.test1_survival = pd.read_csv('../preprocess/test1_preprocess_1.csv')
        self.test1_spent = pd.read_csv('../preprocess/test1_preprocess_2.csv')
        self.test2_survival = pd.read_csv('../preprocess/test2_preprocess_1.csv')
        self.test2_spent = pd.read_csv('../preprocess/test2_preprocess_2.csv')
    
    def get_test1_data(self):
        return self.test1_survival, self.test1_spent
    
    def get_test2_data(self):
        return self.test2_survival, self.test2_spent

class Models:
    def __init__(self):
        self.survival_clf_model = pickle.load(open("../model/final_model_1.sav", "rb"))
        self.survival_reg_model = pickle.load(open("../model/final_model_2.sav", "rb"))
        self.spent_reg_model = pickle.load(open("../model/final_model_3.sav", "rb"))

    def predict(self, st_input_data, as_input_data):
        # survival_time 예측
        # 잔존/비잔존 예측
        X_test_survival = st_input_data.sort_values(by='acc_id').iloc[:, 1:].values
        st_input_data['survive_or_not'] = self.survival_clf_model.predict(X_test_survival)
        
        # 비잔존유저의 생존기간 예측
        test_survival_reg = st_input_data[st_input_data['survive_or_not']==0]
        X_test_survival_reg = test_survival_reg.drop('survive_or_not', axis=1).iloc[:, 1:].values
        st_input_data.loc[st_input_data['survive_or_not']==0, 'survival_time'] = \
            np.clip(np.round(self.survival_reg_model.predict(X_test_survival_reg), 0), 1, 63)
        
        # 잔존유저의 생존기간을 64로 채워주기
        st_input_data['survival_time'].fillna(64.0, inplace=True)
        st_input_data.reset_index(inplace=True)
        st_input_data.drop(['index'], axis=1, inplace=True)
        
        # amount_spent 예측
        X_test_spent_reg = as_input_data.sort_values(by='acc_id').iloc[:, 1:].values
        as_input_data['amount_spent'] = np.clip(self.spent_reg_model.predict(X_test_spent_reg), a_min=0, a_max=None)

        # 예측결과
        result_df = st_input_data[['acc_id', 'survival_time']].sort_values(by='acc_id').copy()
        result_df = pd.merge(result_df, as_input_data[['acc_id', 'amount_spent']], on='acc_id')

        # 필요없는 컬럼 제거
        st_input_data.drop(['survive_or_not', 'survival_time'], axis=1, inplace=True)
        as_input_data.drop(['amount_spent'], axis=1, inplace=True)

        return result_df
    
def save_predict_result(df, f_name):
    df.to_csv(f_name, encoding='utf-8', index=False)

if __name__ == '__main__':
    dataset = DataSet()
    models = Models()

    test1_result = models.predict(*dataset.get_test1_data())
    test2_result = models.predict(*dataset.get_test2_data())

    save_predict_result(test1_result, 'test1_predict.csv')
    save_predict_result(test2_result, 'test2_predict.csv')