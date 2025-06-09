from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
# import subprocess
from kbo_scraper import get_today_games
from db_utils import get_win_probability
from inning_scheduler import start_scheduler
from flask import request
import os
from dotenv import load_dotenv
import threading
import logging
from logging.handlers import RotatingFileHandler
import sys

# 로그 파일 생성 내용이므로 불필요하면 빼도 됨
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "scheduler.log")
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler = RotatingFileHandler(log_file_path, maxBytes=1024*1024, backupCount=5, encoding="utf-8")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


load_dotenv() 

app = Flask(__name__)
scheduler = BackgroundScheduler()
scheduler.start()

import atexit

# Flask 종료될 때 스케줄러 종료
atexit.register(lambda: scheduler.shutdown())

TEAM_CODE_MAP = {
    'LG': 'LG',
    'OB': '두산',
    'SS': '삼성',
    'SK': 'SSG',
    'WO': '키움',
    'KT': 'KT',
    'HH': '한화',
    'HT': 'KIA',
    'NC': 'NC',
    'LT': '롯데'
}
def launch_scheduler_for_game(game_id, start_time):
    def run():
        logger.info(f"[{game_id}] 스케줄 실행됨")
        home_win_pred = 0.5
        # game_id에서 팀명매핑
        away_code = game_id[8:10]
        home_code = game_id[10:12]
        away_team = TEAM_CODE_MAP.get(away_code)
        home_team = TEAM_CODE_MAP.get(home_code)
        if away_team and home_team: # 잘 들어왔을 경우 예측값 db에서 받아오기
            home_win_pred = get_win_probability(home_team, away_team) # 경기 전 예측 값 (team1 == home 일 때 받아오기)
            print(home_win_pred)
            home_win_pred = home_win_pred*0.01
            print("db에서 겟 완")
        logger.info(f"홈팀 승률 예측값: {home_win_pred}")
        start_scheduler(game_id, home_win_pred)

    scheduler.add_job(run, trigger='date', run_date=start_time)
    logger.info(f"[{game_id}] 스케줄 등록 완료 → {start_time}")

def register_today_games():
    today = datetime.now().date()
    logger.info(f"오늘 날짜: {today}")
    games = get_today_games(today)
    logger.info(f"오늘 경기 목록: {games}")

    for game in games:
        try:
            game_time_str = game['start_time']  # '18:30'
            game_time = datetime.strptime(game_time_str, "%H:%M").time()
            full_start = datetime.combine(today, game_time)
            logger.info(f"[{game['game_id']}] 경기 시작 시간: {full_start}")
            launch_scheduler_for_game(game['game_id'], full_start)
        except Exception as e:
            logger.error(f"{game['game_id']} 등록 실패: {e}")

'''
2025-06-03
[{'game_id': '20250603LGNC', 'start_time': datetime.datetime(2025, 6, 3, 14, 0)}, {'game_id': '20250603KTHH', 'start_time': datetime.datetime(2025, 6, 3, 14, 0)}, {'game_id': '20250603HTOB', 'start_time': datetime.datetime(2025, 6, 3, 17, 0)}, {'game_id': '20250603SSSK', 'start_time': datetime.datetime(2025, 6, 3, 17, 0)}, {'game_id': '20250603WOLT', 'start_time': datetime.datetime(2025, 6, 3, 17, 0)}]
'''
# 매일 새벽 3시 예약
# scheduler.add_job(register_today_games, trigger='cron', hour=3, minute=0)
# 렌더는 깨어있지 않대. free 플랜을 쓴다고

@app.route('/health')
def health():
    return jsonify({'jobs': len(scheduler.get_jobs())})

@app.route('/force_register', methods=['GET'])
def force_register():
    token = request.args.get("token")  
    secret_token = os.getenv("REGISTER_SECRET_TOKEN")
    print(f"[DEBUG] 요청 토큰: {token}")
    print(f"[DEBUG] 서버 저장 토큰: {secret_token}")
    if token != secret_token:
        return "Unauthorized", 403
    # register_today_games()
    threading.Thread(target=register_today_games).start() 
    return "강제 경기 등록 완료" # 이 라우터를 깃액션으로 호출하자

@app.route('/')
def home():
    return "KBO 경기 중 승률 예측 API입니다. /force_register (GET) : 수동 스케줄러 호출을 원하시면 사용하세요."

if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=8080)
    # register_today_games() # 크론 잡으로 app.py 을 3시마다 실행시키자 그냥 
    # 크론 잡 유료임 미친 것
    register_today_games()
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host="0.0.0.0", port=port)
    
