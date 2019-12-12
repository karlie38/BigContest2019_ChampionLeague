import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import lightgbm as lgb
#from sklearn.externals import joblib
import pickle

lgb_clf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'num_leaves': 100,
    'max_depth': 20,
    'subsample': 0.5,
    'subsample_freq': 10
}

lgb_reg_params = {
    'boosting_type' : 'rf',
    'learning_rate' : 0.1,
    'n_jobs': -1,
    'n_estimators': 500,
    'num_leaves': 10,
    'max_depth': 10,
    'subsample': 0.5,
    'subsample_freq': 10,
}


class DataSet:
    def __init__(self):
        self.train_survival = pd.read_csv('../preprocess/train_preprocess_1.csv')
        self.train_spent = pd.read_csv('../preprocess/train_preprocess_2.csv')
        self.test1_survival = pd.read_csv('../preprocess/test1_preprocess_1.csv')
        self.test1_spent = pd.read_csv('../preprocess/test1_preprocess_2.csv')
        self.test2_survival = pd.read_csv('../preprocess/test2_preprocess_1.csv')
        self.test2_spent = pd.read_csv('../preprocess/test2_preprocess_2.csv')
    
    def get_train_data(self):
        return self.train_survival, self.train_spent
    
    def get_test1_data(self):
        return self.test1_survival, self.test1_spent
    
    def get_test2_data(self):
        return self.test2_survival, self.test2_spent

class _CustomBaseVoting:
    def __init__(self, estimators):
        self.estimators = estimators
        self.estimator_cnt = len(self.estimators)
    
    def __str__(self):
        return "\n".join(map(str, self.estimators))
    
    def __repr__(self):
        return "\n".join(map(str, self.estimators))
    
    def fit(self, X_train, Y_train, sample_weight=None, sample_percent=0.5, replace=False):
        random_idxs = [np.random.choice(len(X_train), int(len(X_train) * sample_percent), replace=replace)\
                       for _ in range(self.estimator_cnt)]
        if sample_weight is None:
            self.estimators = [model.fit(X_train[random_idxs[i]], Y_train[random_idxs[i]])\
                               for i, model in enumerate(self.estimators)]
        else:
            self.estimators = [model.fit(X_train[random_idxs[i]], Y_train[random_idxs[i]],
                                         sample_weight=sample_weight[random_idxs[i]]) for i, model in enumerate(self.estimators)]
        return self

class CustomVotingClassifier(_CustomBaseVoting):
    def __init__(self, estimators):
        super().__init__(estimators)
    
    def __str__(self):
        return "<VotingClassifier(estimators)>\n" + super().__str__()
    
    def __repr__(self):
        return "<VotingClassifier(estimators)>\n" + super().__repr__()
    
    def fit(self, X_train, Y_train, sample_weight=None, sample_percent=0.5, replace=False):
        return super().fit(X_train, Y_train, sample_weight, sample_percent, replace)
    
    def predict(self, X):
        predict_probas = [model.predict_proba(X) for model in self.estimators]
        pred_test = np.argmax(sum(predict_probas) / self.estimator_cnt, axis=1)
        return pred_test


class CustomVotingRegressor(_CustomBaseVoting):
    def __init__(self, estimators):
        super().__init__(estimators)
    
    def __str__(self):
        return "<VotingRegressor(estimators)>\n" + super().__str__()
    
    def __repr__(self):
        return "<VotingRegressor(estimators)>\n "+ super().__repr__()
    
    def fit(self, X_train, Y_train, sample_weight=None, sample_percent=0.5, replace=False):
        return super().fit(X_train, Y_train, sample_weight, sample_percent, replace)
    
    def predict(self, X):
        predicts = [model.predict(X) for model in self.estimators]
        pred_test = sum(predicts) / self.estimator_cnt
        return pred_test


def make_lgb_models(model, params, model_cnt=6):
    lgb_models = [model(boosting_type=params.get('boosting_type', 'gbdt'),
                        learning_rate=params.get('learning_rate', 0.1),
                        n_jobs=params.get('n_jobs', -1),
                        n_estimators=params.get('n_estimators', 100),
                        num_leaves=params.get('num_leaves', 31),
                        max_depth=params.get('max_depth', -1),
                        bagging_seed=i+1,
                        subsample=params.get('subsample', 1.0),
                        subsample_freq=params.get('subsample_freq', 0)
                        )\
                  for i in range(model_cnt)]
    return lgb_models


def custom_cost(y_true, y_pred):
    """
        model의 object function 정의
        grad, hess는 각각 score_function의 amount_spent_predict에 대한 1차, 2차 미분 값
    """
    grad = (0.18**0.5) * (y_pred - 15*y_true)
    hess = (0.18**0.5) * np.ones(shape=y_true.shape)
    return (grad, hess)

def custom_score(y_pred, y_true):
    """
       survival_time_predict=survival_time_actual 일때 score function 정의
       survival_time_predict와 survival_time_actual은 64일이 아니라는 가정
    """
    score = np.where((y_true>15*y_pred)|(y_true==0),
                     -0.3*y_pred,
                     np.where(y_true<=y_pred,
                              30*y_true-0.3*y_pred,
                              y_pred*(2973/90)-(30/9)*y_true))
    return ("custom_score", np.sum(score), True)

def train_model(train_survival, train_spent):
    # survival time 학습
    # 잔존/비잔존 분류 학습을 위한 데이터
    X_train_survival = train_survival.iloc[:, 1:-2].values
    Y_train_survival = train_survival['survival_time'].map(lambda x: 1 if x==64 else 0).values

    # 분류를 위한 앙상블의 앙상블 학습
    clf_survive_models = make_lgb_models(lgb.LGBMClassifier, lgb_clf_params, 6)
    st_custom_voting_clf = CustomVotingClassifier(clf_survive_models)
    st_custom_voting_clf = st_custom_voting_clf.fit(X_train_survival, Y_train_survival)

    # 학습시킬 비잔존 유저의 데이터 추출
    train_survival_reg = train_survival[train_survival['survival_time'] != 64]
    
    # 비잔존 유저의 잔존일수 회귀 학습을 위한 데이터
    X_train_survival_reg = train_survival_reg.iloc[:, 1:-2].values
    Y_train_survival_reg = train_survival_reg['survival_time'].values
    Y_train_survival_reg_forWeight = np.e * train_survival_reg['amount_spent'].values

    # 회귀를 위한 앙상블의 앙상블 학습
    reg_survive_models = make_lgb_models(lgb.LGBMRegressor, lgb_reg_params, 6)
    st_custom_voting_reg = CustomVotingRegressor(reg_survive_models)
    st_custom_voting_reg = st_custom_voting_reg.fit(X_train_survival_reg, Y_train_survival_reg,
                                                    sample_weight=Y_train_survival_reg_forWeight)

    # amount spent 학습
    # train feature data, train label data
    X = train_spent.iloc[:, 1:-2].values
    Y = train_spent['amount_spent'].values
    
    # 회귀를 위한 LightGBM 앙상블 모델 학습(custom objective를 이용)
    as_lgb_reg = lgb.LGBMRegressor(num_leaves=100, max_depth=20, n_estimators=20)
    as_lgb_reg.set_params(**{'objective': custom_cost})

    X_train_spent, X_test_spent, Y_train_spent, Y_test_spent = train_test_split(X, Y, test_size = 0.3, random_state=42)
    
    as_lgb_reg = as_lgb_reg.fit(X_train_spent, Y_train_spent,
                  eval_set=[(X_test_spent, Y_test_spent)],
                  eval_metric=custom_score,
                  verbose=0)
    
    return st_custom_voting_clf, st_custom_voting_reg, as_lgb_reg

def save_model(model, f_name):
    #joblib.dump(model, f_name)
    pickle.dump(model, open(f_name, 'wb'))


if __name__ == '__main__':
    dataset = DataSet()
    
    survival_clf, survival_reg, spent_reg = train_model(*dataset.get_train_data())
    save_model(survival_clf, 'final_model_1.sav')
    save_model(survival_reg, 'final_model_2.sav')
    save_model(spent_reg, 'final_model_3.sav')