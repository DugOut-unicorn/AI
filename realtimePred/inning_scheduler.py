import schedule 
import time
# import subprocess
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from realtime_crawling.get_log import get_driver
from inference import inference

prev_inning = None

def get_current_inning(game_id: str) -> int:
    driver =  get_driver() #webdriver.Chrome()
    url = f"https://www.koreabaseball.com/Game/LiveText.aspx?leagueId=1&seriesId=0&gameId={game_id}0&gyear=2025"
    driver.get(url)

    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.ID, "tblScoreBoard2")))

    max_inning = 12
    inning_done = 0

    try:
        for i in range(1, max_inning + 1):
            away_td = driver.find_element(By.ID, f"rptScoreBoard2_tdInn{i}_0")  # 어웨이팀
            home_td = driver.find_element(By.ID, f"rptScoreBoard2_tdInn{i}_1")  # 홈팀
            if '-' in (away_td.text.strip(), home_td.text.strip()):
                break
            inning_done = i
    except Exception as e:
        print(f"⚠️ 이닝 정보 파싱 중 오류: {e}")
        driver.quit()
        return None 

    driver.quit()
    return inning_done

def run_inference(inning: int, game_id: str, home_win_pred: float):
    print(f"\n[{inning}회 종료] 추론 시작 (game_id={game_id})")
    # subprocess.run(['python', 'inference.py', '--inning', str(inning), '--game_id', game_id, '--home_win_pred', str(home_win_pred)])
    inference(inning=inning, game_id=game_id, home_win_pred=home_win_pred)

# 경기 시작 후 이닝 끝남 감지
def run_inference_if_inning_finished(game_id: str, home_win_pred: float):
    global prev_inning
    try:
        current_inning = get_current_inning(game_id)

        if current_inning is None:
            print("이닝 정보를 얻지 못함. 이전 값 유지")
            return

        if prev_inning is not None and current_inning > prev_inning:
            run_inference(prev_inning, game_id, home_win_pred)

        prev_inning = current_inning
        print(f"현재 종료된 이닝: {current_inning}회")

    except Exception as e:
        print(f"예외 발생: {e}")

# 스케줄러 시작 함수
def start_scheduler(game_id: str, home_win_pred: float):
    print(f"실시간 추론 스케줄 시작... (game_id={game_id})")
    schedule.every(100).seconds.do(run_inference_if_inning_finished, game_id=game_id, home_win_pred=home_win_pred)


    while True:
        schedule.run_pending()
        time.sleep(1)
