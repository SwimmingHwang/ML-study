"""
perceptron : 가장 간단한 인공 신경망 구조
TLU threshold logic unit : 퍼셉트론은 TLU 인공 뉴련을 기반으로 하며 입력의 가중치 합을 계산한 뒤,
계산된 합에 계단함수를 적용하여 결과를 출력
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:,(2,3)]
y = (iris.target == 0).astype(np.int)

per_Clf = Perceptron()
per_Clf.fit(X, y)

y_pred = per_Clf.predict(X)
print(y_pred)
'''
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0]
'''