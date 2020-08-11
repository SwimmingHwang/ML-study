"""
[앙상블]
로지스틱 회기 분류기, 랜덤 포레스트 분류기, SVM 분류기를 조합하여 투표 기반 분류기 생성 및 훈련 예제
"""
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 마주보는 두 개의 반원 모양으로 데이터 포인터가 놓여 있는 이진 분류를 위한 데이터 셋
X, y = make_moons(n_samples=10000, noise=0.15)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

# 사이킷런의 투표 기반 분류기 생성
# voting : 직접투표 # 각 분류기의 예측을 모아서 가장 많이 선택도니 클래스를 예측하는 것(다수결 투표)
voting_clf = VotingClassifier(
    estimators=[('lr',log_clf), ('rf', rnd_clf),('svc',svm_clf)],
    voting='hard'
)
voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))