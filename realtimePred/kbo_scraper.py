import requests
from datetime import datetime, date

def get_today_games(today_date):
    print(today_date.strftime("%Y-%m-%d"))
    session = requests.Session()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx"
    }

    data = {
        "leId": "1",
        "srId": "0,1,3,4,5,7,9",
        "date": today_date.strftime("%Y%m%d")
    }

    response = session.post("https://www.koreabaseball.com/ws/Main.asmx/GetKboGameList", headers=headers, data=data)
    if response.status_code != 200:
        print("❌ API 요청 실패:", response.status_code)
        return []

    json_data = response.json()
    if "game" not in json_data:
        print("❌ 'game' 키 없음")
        return []

    games = []
    for game in json_data["game"]:
        games.append({
            "game_id": game.get("G_ID", "")[:12],
            "away": game.get("AWAY_NM", ""),
            "home": game.get("HOME_NM", ""),
            "stadium": game.get("S_NM", "알 수 없음"),
            "start_time": game.get("G_TM", "00:00"),
            "status": game.get("GAME_SC_NM", "알 수 없음")  # 경기예정/종료 등
        })

    return games