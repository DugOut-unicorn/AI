# 종엽님 코드 그대로 가져온 거에 추가
# 나중에 수정할지도 
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import datetime

# .env 파일 로드는 이 스크립트가 독립적으로 실행될 때 필요할 수 있습니다.
# GitHub Actions에서는 Secrets로 환경 변수를 직접 주입하므로 필수는 아니지만,
# 로컬 개발 및 테스트를 위해 남겨두는 것이 좋습니다.
load_dotenv()

def get_db_engine():
    """DB 연결을 위한 SQLAlchemy 엔진 생성"""
    # 환경 변수 'DB_URI'에서 DB 연결 문자열을 가져옵니다.
    # 환경 변수가 설정되지 않았을 경우를 대비하여 기본값(하드코딩된 값)을 제공합니다.
    # 이 기본값은 개발 환경에서만 사용하고, 프로덕션에서는 환경 변수를 통해 주입되어야 합니다.
    db_uri = os.environ.get("DB_URI")

    # 디버깅을 위해 DB URI 사용 여부를 출력합니다.
    # 보안을 위해 비밀번호는 출력하지 않도록 마스킹 처리합니다.
    try:
        parts = db_uri.split('@')
        if len(parts) > 1:
            user_host = parts[0]
            db_path = parts[1]
            user_parts = user_host.split('//')[1].split(':')
            masked_user_host = user_host.split('//')[0] + '//' + user_parts[0] + ':****'
            print(f"DEBUG: Using DB URI: {masked_user_host}@{db_path}")
        else:
            print(f"DEBUG: Using DB URI (masking failed): {db_uri}")
    except Exception as e:
        print(f"DEBUG: Could not mask DB URI for logging: {e}")
        print(f"DEBUG: Using DB URI: {db_uri}")


    if not db_uri:
        raise ValueError("DB URI 환경 변수 (DB_URI)가 설정되지 않았습니다.")

    engine = create_engine(db_uri)
    return engine

# 종엽님코드 그대로 가져온 후 save함수 4개 지우고 내 거 이어씀. 코드 합칠 때 참고

def save_live_win_prediction(game_id: str, inning: int, win_prob: float, home_accum_score: int, away_accum_score: int):
    """경기 중 실시간 예측 결과를 DB에 저장"""
    try:
        engine = get_db_engine()
        table_name = 'live_win_predictions'

        # 저장할 데이터 구성 
        # home_win_pred, inning, game_id, home_score, away_score
        df = pd.DataFrame([{
            'game_id': game_id,
            'inning': inning,
            'win_probability': win_prob,
            'home_accum_score' : home_accum_score,
            'away_accum_score' : away_accum_score,
            'predicted_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])

        # 저장
        df.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"실시간 예측 결과 저장 완료: {game_id}, {inning}회, {win_prob:.3f}")
    except Exception as e:
        print(f"실시간 예측 결과 DB 저장 중 오류 발생: {e}")


# 경기 전 예측 값 (win_probabilities 테이블에서 team1 == home 일 때 받아오기) 받아오기

def get_win_probability(team1: str, team2: str) -> float:
    """win_probabilities 테이블에서 team1 vs team2의 승률 가져오기"""
    try:
        engine = get_db_engine()
        query = text("""
            SELECT win_probability 
            FROM win_probabilities 
            WHERE team1 = :team1 AND team2 = :team2 
            ORDER BY prediction_date DESC 
            LIMIT 1;
        """)
        with engine.connect() as conn:
            result = conn.execute(query, {"team1": team1, "team2": team2}).fetchone()
            if result:
                return float(result[0])
            else:
                print(f"해당 팀 조합({team1} vs {team2})에 대한 데이터가 없습니다.")
                return None
    except Exception as e:
        print(f"승률 조회 중 오류 발생: {e}")
        return None


def delete_predictions_by_game_id(game_id: str):
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM live_win_predictions WHERE game_id = :game_id"), {"game_id": game_id})
            conn.commit()
        print(f"삭제 완료: game_id={game_id}")
    except Exception as e:
        print(f"삭제 실패: {e}")

def delete_all_predictions():
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM live_win_predictions"))
            conn.commit()
        print("전체 예측 데이터 삭제 완료")
    except Exception as e:
        print(f"전체 삭제 실패: {e}")
        
# delete_all_predictions()
