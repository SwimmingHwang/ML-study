
#
#  가장 중요한 특성을 선택하는 변환기를 준비 파이프라인에 추가
#

import ch2_ex1_SVR_tuning

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

housing = ch2_ex1_SVR_tuning.housing
housing_prepared = ch2_ex1_SVR_tuning.housing_prepared
housing_labels = ch2_ex1_SVR_tuning.housing_labels
grid_search = ch2_ex1_SVR_tuning.grid_search
attributes = ch2_ex1_SVR_tuning.attributes
full_pipeline = ch2_ex1_SVR_tuning.full_pipeline

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]


k = 5
feature_importances = grid_search.best_estimator_.feature_importances_
top_k_feature_indices = indices_of_top_k(feature_importances, k)
print(top_k_feature_indices) # array([ 0,  1,  7,  9, 12])

print(sorted(zip(feature_importances, attributes), reverse=True)[:k])
'''
[(0.36615898061813423, 'median_income'),
 (0.16478099356159054, 'INLAND'),
 (0.10879295677551575, 'pop_per_hhold'),
 (0.07334423551601243, 'longitude'),
 (0.06290907048262032, 'latitude')]
 '''

# 파이프라인 생성
preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])

housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)
# 최상의 k개 특성 맞는지 확인
print(housing_prepared[0:3, top_k_feature_indices])
'''
array([[-1.15604281,  0.77194962, -0.61493744, -0.08649871,  0.        ],
       [-1.17602483,  0.6596948 ,  1.33645936, -0.03353391,  0.        ],
       [ 1.18684903, -1.34218285, -0.5320456 , -0.09240499,  0.        ]])
'''

print(housing_prepared_top_k_features[0:3])


#
# 전체 데이터 준비과정 & 최종 예측 파이프라인 생성
#

prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('svm_reg', SVR(**grid_search.best_params_))
])
prepare_select_and_predict_pipeline.fit(housing, housing_labels)

some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))

#
# GridSearchCV 준비 단계 옵션 서치 자동화
#
from sklearn.model_selection import GridSearchCV

param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
                                scoring='neg_mean_squared_error', verbose=2)
grid_search_prep.fit(housing, housing_labels)