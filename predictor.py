import pandas as pd
import numpy as np
import datetime
from crawler import crawl_hitter_data, crawl_pitcher_data, load_historical_data
from data_processor import process_hitter_data, process_pitcher_data


def generate_win_probability_df(all_hitter_data, all_pitcher_data):
    """팀별 승률 계산"""
    # 연도가 2025인 데이터만 추출
    df_2025_hitter = all_hitter_data[all_hitter_data['연도'] == 2025]
    df_2025_pitcher = all_pitcher_data[all_pitcher_data['연도'] == 2025]

    # 팀별 OPS 값 평균 산출
    team_ops_avg = df_2025_hitter.groupby('팀명')['OPS_predict'].mean().reset_index()

    # OPS 기준 내림차순으로 팀 정렬
    team_ops_avg_sorted = team_ops_avg.sort_values(by='OPS_predict', ascending=False)

    # 팀별 WHIP 값 평균 산출
    team_whip_avg = df_2025_pitcher.groupby('팀명')['WHIP_predict'].mean().reset_index()

    # WHIP 기준 내림차순으로 팀 정렬
    team_whip_avg_sorted = team_whip_avg.sort_values(by='WHIP_predict', ascending=False)

    # 두 데이터 병합
    victory_predict_df = pd.merge(team_whip_avg_sorted, team_ops_avg_sorted, on='팀명')
    victory_predict_df['OPS_minus_WHIP'] = victory_predict_df['OPS_predict'] - victory_predict_df['WHIP_predict']

    # OPS_minus_WHIP 기준 내림차순으로 팀 정렬
    victory_predict_df = victory_predict_df.sort_values(by='OPS_minus_WHIP', ascending=False)

    # OPS_minus_WHIP의 최솟값 절댓값 계산
    min_ops_minus_whip = victory_predict_df['OPS_minus_WHIP'].min()
    adjustment_value = abs(min_ops_minus_whip) + 0.1

    # 조정된 점수 계산
    victory_predict_df['Adjusted_Score'] = victory_predict_df['OPS_minus_WHIP'] + adjustment_value

    # 팀 목록 추출
    teams = victory_predict_df['팀명'].tolist()

    # 승률 결과를 저장할 데이터프레임 초기화
    win_probability_df = pd.DataFrame(index=teams, columns=teams)

    # 팀 간 승률 계산 및 데이터프레임 채우기
    for team_a in teams:
        score_a = victory_predict_df[victory_predict_df['팀명'] == team_a]['Adjusted_Score'].iloc[0]
        for team_b in teams:
            if team_a == team_b:
                win_probability_df.loc[team_a, team_b] = '-'  # 같은 팀 간 승률은 의미 없음
            else:
                score_b = victory_predict_df[victory_predict_df['팀명'] == team_b]['Adjusted_Score'].iloc[0]
                # 승리 확률 계산: (Sa / (Sa + Sb)) * 100%
                win_prob = (score_a / (score_a + score_b)) * 100
                win_probability_df.loc[team_a, team_b] = round(win_prob, 2)  # 소수점 둘째 자리까지 반올림

    return win_probability_df


def get_win_probability_df(cached_data):
    """승률 데이터를 가져오거나 캐시된 데이터 사용"""
    current_time = datetime.datetime.now()

    # 매일 00:00~00:04 사이 강제 업데이트
    if current_time.hour == 0 and current_time.minute < 5:
        print("⚠️ 일일 강제 업데이트 시간(00:00) - 캐시 초기화")
        cached_data['last_update'] = None

    # 24시간(86,400초) 지났는지 확인
    if (cached_data['win_probability_df'] is None or
            cached_data['last_update'] is None or
            (current_time - cached_data['last_update']).total_seconds() > 86400):

        print("🔁 데이터 새로고침 시작...")

        # 크롤링 수행
        hitter_data_2025 = crawl_hitter_data()
        pitcher_data_2025 = crawl_pitcher_data()

        # 역대 데이터 로드
        hitter_data_his, pitcher_data_his = load_historical_data()

        # 데이터 처리
        all_hitter_data = process_hitter_data(hitter_data_2025, hitter_data_his)
        all_pitcher_data = process_pitcher_data(pitcher_data_2025, pitcher_data_his)

        # 승률 계산
        win_probability_df = generate_win_probability_df(all_hitter_data, all_pitcher_data)

        # 캐시 업데이트
        cached_data['hitter_data'] = all_hitter_data
        cached_data['pitcher_data'] = all_pitcher_data
        cached_data['win_probability_df'] = win_probability_df
        cached_data['last_update'] = current_time

        # 다음 업데이트 시간 계산
        next_update = current_time + datetime.timedelta(hours=24)
        print(f"✅ 데이터 새로고침 완료! 다음 업데이트: {next_update.strftime('%Y-%m-%d %H:%M')}")
    else:
        # 남은 시간 계산
        remaining_time = 86400 - (current_time - cached_data['last_update']).total_seconds()
        print(f"💾 캐시된 데이터 사용 (남은 시간: {round(remaining_time / 3600, 1)}시간)")

    return cached_data['win_probability_df']
