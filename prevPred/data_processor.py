import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def process_hitter_data(hitter_data_2025, hitter_data_his):
    """타자 데이터 처리"""
    # 현재 데이터와 역대 데이터 합치기
    all_hitter_data = pd.concat([hitter_data_his, hitter_data_2025], ignore_index=True)

    # '순위', '선수명', '팀명' 컬럼을 제외한 나머지 컬럼을 실수형으로 변환
    columns_to_convert = all_hitter_data.columns.difference(['선수명', '팀명', '연도'])

    # 실수형으로 변환
    all_hitter_data[columns_to_convert] = all_hitter_data[columns_to_convert].astype(float)

    # OPS 계산
    all_hitter_data['OBP'] = (all_hitter_data['H'] + all_hitter_data['RBI'] + all_hitter_data['SAC']) / (
                all_hitter_data['AB'] + all_hitter_data['RBI'] + all_hitter_data['SAC'] + all_hitter_data['SF'])
    all_hitter_data['SLG'] = all_hitter_data['TB'] / all_hitter_data['AB']
    all_hitter_data['OPS'] = all_hitter_data['OBP'] + all_hitter_data['SLG']

    # 선수명, 순위, 팀명 열 제거
    X = all_hitter_data.drop(columns=['선수명', '팀명', '연도'])

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA 수행
    pca = PCA()
    pca.fit(X_scaled)

    # 주성분의 분산 비율
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    # 최적의 주성분 개수 판단 (90% 누적 분산 비율)
    n = next(i for i, total in enumerate(cumulative_variance) if total >= 0.90) + 1

    # 데이터 프레임에서 종속 변수와 독립 변수 분리
    y = X['OPS']
    X = X.drop(columns=['OPS', 'SLG', 'OBP'])

    # 데이터 분할 (훈련 세트와 테스트 세트)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 생성 및 훈련
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 변수 중요도 추출
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 변수 중요도 높은 순서대로 n개 까지 냅두고 다른 열은 삭제
    top_n_features = X.columns[indices][:n]
    X_reduced = X[top_n_features]

    sc = StandardScaler()
    df_scaled = sc.fit_transform(X_reduced)
    df_scaled_df = pd.DataFrame(df_scaled, columns=X_reduced.columns)
    df_c = df_scaled_df.dropna()

    # 군집 개수 범위 설정
    range_n_clusters = range(2, 11)

    # Elbow method와 실루엣 계수 저장을 위한 리스트 초기화
    inertia = []
    silhouette_scores = []

    # KMeans 클러스터링 수행
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(df_c)

        # inertia (Elbow method)
        inertia.append(kmeans.inertia_)

        # 실루엣 계수
        if n_clusters > 1:
            score = silhouette_score(df_c, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)

    # 최적의 군집 개수 찾기
    optimal_k_elbow = np.diff(inertia, 2).argmin() + 2
    optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores[1:])) + 2

    # 두 지표의 평균을 사용하여 최적의 군집 개수 결정
    average_scores = [(inertia[i] + silhouette_scores[i - 1]) / 2 for i in range(1, len(inertia))]
    optimal_k_combined = average_scores.index(min(average_scores)) + 2

    # 최적의 군집 개수를 변수 k에 저장
    k = optimal_k_combined

    kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++')
    clusters = kmeans.fit(df_c)

    # 클러스터링 결과 저장
    df_c['cluster'] = clusters.labels_
    df_c['선수명'] = all_hitter_data['선수명']
    df_c['팀명'] = all_hitter_data['팀명']
    df_c['OPS'] = all_hitter_data['OPS']

    df_r = df_c.drop(columns=['선수명', '팀명'])

    X = df_r.drop(['OPS'], axis=1)
    y = df_r['OPS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 정의
    models = {
        'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # 최적의 모델 학습
    best_model = models['Random Forest Regression']
    best_model.fit(X_train, y_train)

    # 전체 데이터에 대한 예측
    X_full = df_r.drop(['OPS'], axis=1)
    y_full_pred = best_model.predict(X_full)

    # 예측 결과를 DataFrame으로 변환
    predictions_df = pd.DataFrame({
        'Actual OPS': df_r['OPS'],
        'Predicted OPS': y_full_pred
    })

    all_hitter_data['OPS_predict'] = predictions_df['Predicted OPS']

    return all_hitter_data


def process_pitcher_data(pitcher_data_2025, pitcher_data_his):
    """투수 데이터 처리"""
    # 현재 데이터와 역대 데이터 합치기
    all_pitcher_data = pd.concat([pitcher_data_his, pitcher_data_2025], ignore_index=True)

    # 'IP' 컬럼을 실수형으로 변환하는 함수 정의
    def convert_ip_to_float(ip_str):
        if isinstance(ip_str, (int, float)):
            return float(ip_str)
        ip_str = str(ip_str).strip()
        if ' ' in ip_str:
            parts = ip_str.split(' ')
            whole = float(parts[0])
            numerator, denominator = map(float, parts[1].split('/'))
            return whole + numerator / denominator
        elif '/' in ip_str:
            numerator, denominator = map(float, ip_str.split('/'))
            return numerator / denominator
        else:
            return float(ip_str)

    # 'IP' 컬럼 변환
    all_pitcher_data['IP'] = all_pitcher_data['IP'].apply(convert_ip_to_float)

    # '순위', '선수명', '팀명' 컬럼을 제외한 나머지 컬럼을 실수형으로 변환
    columns_to_convert = all_pitcher_data.columns.difference(['선수명', '팀명', '연도'])

    # 실수형으로 변환
    all_pitcher_data[columns_to_convert] = all_pitcher_data[columns_to_convert].astype(float)

    # 선수명, 순위, 팀명 열 제거
    X = all_pitcher_data.drop(columns=['선수명', '팀명', '연도'])

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA 수행
    pca = PCA()
    pca.fit(X_scaled)

    # 주성분의 분산 비율
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    # 최적의 주성분 개수 판단 (90% 누적 분산 비율)
    n = next(i for i, total in enumerate(cumulative_variance) if total >= 0.90) + 1

    # 데이터 프레임에서 종속 변수와 독립 변수 분리
    y = X['WHIP']
    X = X.drop(columns=['WHIP', 'H', 'BB', 'IP'])

    # 데이터 분할 (훈련 세트와 테스트 세트)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 생성 및 훈련
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 변수 중요도 추출
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 변수 중요도 높은 순서대로 n개 까지 냅두고 다른 열은 삭제
    top_n_features = X.columns[indices][:n]
    X_reduced = X[top_n_features]

    sc = StandardScaler()
    df_scaled = sc.fit_transform(X_reduced)
    df_scaled_df = pd.DataFrame(df_scaled, columns=X_reduced.columns)
    df_c = df_scaled_df.dropna()

    # 군집 개수 범위 설정
    range_n_clusters = range(2, 11)

    # Elbow method와 실루엣 계수 저장을 위한 리스트 초기화
    inertia = []
    silhouette_scores = []

    # KMeans 클러스터링 수행
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(df_c)

        # inertia (Elbow method)
        inertia.append(kmeans.inertia_)

        # 실루엣 계수
        if n_clusters > 1:
            score = silhouette_score(df_c, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)

    # 최적의 군집 개수 찾기
    optimal_k_elbow = np.diff(inertia, 2).argmin() + 2
    optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores[1:])) + 2

    # 두 지표의 평균을 사용하여 최적의 군집 개수 결정
    average_scores = [(inertia[i] + silhouette_scores[i - 1]) / 2 for i in range(1, len(inertia))]
    optimal_k_combined = average_scores.index(min(average_scores)) + 2

    # 최적의 군집 개수를 변수 k에 저장
    k = optimal_k_combined

    kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++')
    clusters = kmeans.fit(df_c)

    # 클러스터링 결과 저장
    df_c['cluster'] = clusters.labels_
    df_c['선수명'] = all_pitcher_data['선수명']
    df_c['팀명'] = all_pitcher_data['팀명']
    df_c['WHIP'] = all_pitcher_data['WHIP']

    df_r = df_c.drop(columns=['선수명', '팀명'])

    X = df_r.drop(['WHIP'], axis=1)
    y = df_r['WHIP']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 정의
    models = {
        'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # 최적의 모델 학습
    best_model = models['Random Forest Regression']
    best_model.fit(X_train, y_train)

    # 전체 데이터에 대한 예측
    X_full = df_r.drop(['WHIP'], axis=1)
    y_full_pred = best_model.predict(X_full)

    # 예측 결과를 DataFrame으로 변환
    predictions_df = pd.DataFrame({
        'Actual WHIP': df_r['WHIP'],
        'Predicted WHIP': y_full_pred
    })

    all_pitcher_data['WHIP_predict'] = predictions_df['Predicted WHIP']

    return all_pitcher_data
