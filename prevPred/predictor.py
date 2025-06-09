# predictor.py
import pandas as pd
import numpy as np
import datetime
from crawler import crawl_hitter_data, crawl_pitcher_data, load_historical_data
from data_processor import process_hitter_data, process_pitcher_data


def generate_win_probability_df(all_hitter_data, all_pitcher_data):
    """팀별 승률 계산 및 OPS-WHIP 기반 팀 순위 예측"""
    # 연도가 2025인 데이터만 추출
    df_2025_hitter = all_hitter_data[all_hitter_data['연도'] == 2025]
    df_2025_pitcher = all_pitcher_data[all_pitcher_data['연도'] == 2025]

    # --- Debugging (can remove after fix) ---
    # print("\n--- Debugging in generate_win_probability_df ---")
    # print("df_2025_hitter columns:", df_2025_hitter.columns.tolist())
    # print("df_2025_pitcher columns:", df_2025_pitcher.columns.tolist())
    # --- End Debugging ---


    # 팀별 OPS 값 평균 산출
    team_ops_avg = df_2025_hitter.groupby('팀명')['OPS_predict'].mean().reset_index()

    # OPS 기준 내림차순으로 팀 정렬
    team_ops_avg_sorted = team_ops_avg.sort_values(by='OPS_predict', ascending=False)

    # 팀별 WHIP 값 평균 산출
    team_whip_avg = df_2025_pitcher.groupby('팀명')['WHIP_predict'].mean().reset_index()

    # WHIP 기준 내림차순으로 팀 정렬
    team_whip_avg_sorted = team_whip_avg.sort_values(by='WHIP_predict', ascending=False)

    # --- Debugging (can remove after fix) ---
    # print("team_ops_avg columns:", team_ops_avg.columns.tolist())
    # print("team_whip_avg columns:", team_whip_avg.columns.tolist())
    # --- End Debugging ---

    # --- 순위 예측 로직 (기존 유지) ---
    team_stats = pd.merge(team_ops_avg, team_whip_avg, on='팀명') # suffixes 제거 유지
    team_stats['OPS_minus_WHIP'] = team_stats['OPS_predict'] - team_stats['WHIP_predict']
    team_rankings = team_stats.sort_values(by='OPS_minus_WHIP', ascending=False).reset_index(drop=True)
    team_rankings['rank'] = team_rankings.index + 1
    predicted_team_rankings_df = team_rankings[['rank', '팀명', 'OPS_predict', 'WHIP_predict', 'OPS_minus_WHIP']]
    predicted_team_rankings_df.columns = ['rank', 'team_name', 'predicted_ops', 'predicted_whip', 'ops_minus_whip']
    # --- 순위 예측 로직 끝 ---


    # --- 승률 예측 로직 (새로운 로직 적용) ---
    # 두 데이터 병합 (새로운 승률 로직에서는 OPS_predict와 WHIP_predict가 합쳐진 새로운 DF 사용)
    # victory_predict_df는 OPS_minus_WHIP 계산에 사용되지만,
    # OPS_predict와 WHIP_predict는 각각의 원래 컬럼 이름으로 merge되어야 함.
    # 이전 오류를 방지하기 위해 merge 시 suffixes 제거.
    victory_predict_df = pd.merge(team_whip_avg, team_ops_avg, on='팀명') # suffixes 제거
    victory_predict_df['OPS_minus_WHIP'] = victory_predict_df['OPS_predict'] - victory_predict_df['WHIP_predict']

    # OPS_minus_WHIP 기준 내림차순으로 팀 정렬
    victory_predict_df = victory_predict_df.sort_values(by='OPS_minus_WHIP', ascending=False)

    # OPS_minus_WHIP의 최솟값 절댓값 계산
    min_ops_minus_whip = victory_predict_df['OPS_minus_WHIP'].min()
    # 음수 값을 양수로 변환하여 조정치를 확보. 0.1은 0으로 나뉘는 것을 방지
    adjustment_value = abs(min_ops_minus_whip) + 0.1

    # 조정된 점수 계산 (모든 점수를 양수로 만듦)
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
                # 분모가 0이 되는 경우를 방지 (score_a + score_b가 0이 되는 경우)
                if (score_a + score_b) == 0:
                    win_prob = 50.0 # 0으로 나누는 경우를 대비하여 기본값 설정 (예: 50%)
                else:
                    win_prob = (score_a / (score_a + score_b)) * 100
                win_probability_df.loc[team_a, team_b] = round(win_prob, 2)  # 소수점 둘째 자리까지 반올림
    # --- 승률 예측 로직 끝 ---


    # generate_win_probability_df 함수는 여전히 두 개의 DataFrame을 반환해야 합니다.
    return win_probability_df, predicted_team_rankings_df


def get_win_probability_df(cached_data):
    """승률 데이터를 가져오거나 캐시된 데이터 사용"""
    current_time = datetime.datetime.now()

    # Render 배포 시 초기 데이터 로드를 위해, 만약 캐시가 비어있다면 강제로 새로고침
    if cached_data['hitter_data'] is None or cached_data['pitcher_data'] is None:
        print("Initial data load for Render deployment: Forcing data refresh.")
        cached_data['last_update'] = None # 캐시 업데이트 시간을 초기화하여 새로고침을 강제합니다.

    # 24시간(86,400초) 지났는지 확인
    # 스케줄러가 매일 00:01에 실행되므로, 이 강제 업데이트 로직은 필요 없을 수 있으나,
    # API 요청 시에도 최신 데이터를 보장하기 위해 유지하는 것이 좋습니다.
    # 다만, tasks.py의 스케줄러와 중복될 수 있으니, 스케줄러가 정확히 작동한다면 이 부분은 더 간결하게 수정 가능합니다.
    # 현재는 '매일 00:00~00:04 사이 강제 업데이트' 로직이 있으므로 그대로 따릅니다.
    if current_time.hour == 0 and current_time.minute < 5:
        print("⚠️ 일일 강제 업데이트 시간(00:00) - 캐시 초기화")
        cached_data['last_update'] = None # 캐시 업데이트 시간을 초기화하여 새로고침을 강제합니다.

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

        # 승률 계산 및 팀 순위 예측 (두 개 모두 반환되도록)
        win_probability_df, predicted_team_rankings_df = generate_win_probability_df(all_hitter_data, all_pitcher_data)

        # 캐시 업데이트
        cached_data['hitter_data'] = all_hitter_data
        cached_data['pitcher_data'] = all_pitcher_data
        cached_data['win_probability_df'] = win_probability_df
        cached_data['predicted_team_rankings_df'] = predicted_team_rankings_df # 추가된 예측 순위 저장
        cached_data['last_update'] = current_time

        # 다음 업데이트 시간 계산 (로그 메시지에만 사용)
        next_update = current_time + datetime.timedelta(hours=24)
        print(f"✅ 데이터 새로고침 완료! 다음 업데이트: {next_update.strftime('%Y-%m-%d %H:%M')}")
    else:
        # 남은 시간 계산 (로그 메시지에만 사용)
        remaining_time = 86400 - (current_time - cached_data['last_update']).total_seconds()
        print(f"💾 캐시된 데이터 사용 (남은 시간: {round(remaining_time / 3600, 1)}시간)")

    return cached_data['win_probability_df']

def get_predicted_team_rankings_df(cached_data):
    """캐시된 팀 순위 예측 DataFrame을 반환"""
    # get_win_probability_df 함수가 호출될 때 이미 데이터가 업데이트되었으므로,
    # 여기서는 단순히 캐시된 데이터를 반환합니다.
    if 'predicted_team_rankings_df' not in cached_data or cached_data['predicted_team_rankings_df'] is None:
        print("⏳ 팀 순위 예측 데이터가 캐시되지 않았습니다. 데이터를 새로고침합니다.")
        get_win_probability_df(cached_data)
    return cached_data['predicted_team_rankings_df']
