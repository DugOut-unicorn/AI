# tasks.py
from crawler import crawl_hitter_data, crawl_pitcher_data, load_historical_data
from data_processor import process_hitter_data, process_pitcher_data
from predictor import generate_win_probability_df
from db_utils import save_win_probabilities, save_team_rankings, save_hitter_data, save_pitcher_data # save_hitter_data, save_pitcher_data 임포트 추가 필요
import datetime

def run_daily_prediction_job():
    """매일 실행될 예측 및 DB 저장 작업"""
    print(f"⏰ {datetime.datetime.now()}: 일일 예측 및 DB 적재 작업을 시작합니다.")

    try:
        # 1. 데이터 크롤링
        hitter_data_2025 = crawl_hitter_data()
        pitcher_data_2025 = crawl_pitcher_data()

        # 2. 역대 데이터 로드
        hitter_data_his, pitcher_data_his = load_historical_data()

        # 3. 데이터 처리
        all_hitter_data = process_hitter_data(hitter_data_2025, hitter_data_his)
        all_pitcher_data = process_pitcher_data(pitcher_data_2025, pitcher_data_his)

        # 4. 승률 예측 및 팀 순위 예측
        # generate_win_probability_df는 이제 두 개의 DataFrame을 반환합니다.
        win_probability_df, predicted_team_rankings_df = generate_win_probability_df(all_hitter_data, all_pitcher_data)

        # 5. DB 저장
        # db_utils에 save_hitter_data와 save_pitcher_data 함수가 있다고 가정하고 호출합니다.
        save_hitter_data(all_hitter_data)
        save_pitcher_data(all_pitcher_data)
        save_win_probabilities(win_probability_df)
        save_team_rankings(predicted_team_rankings_df)

        print(f"✅ {datetime.datetime.now()}: 일일 예측 및 DB 적재 작업이 성공적으로 완료되었습니다.")

    except Exception as e:
        print(f"❌ {datetime.datetime.now()}: 일일 예측 및 DB 적재 작업 중 치명적인 오류 발생: {e}")
        # GitHub Actions가 실패했음을 명확히 하기 위해 예외를 다시 발생시킵니다.
        raise

# 이 부분이 핵심입니다: 스크립트가 직접 실행될 때 run_daily_prediction_job 함수를 호출합니다.
if __name__ == "__main__":
    run_daily_prediction_job()
