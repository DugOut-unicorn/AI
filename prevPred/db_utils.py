# db_utils.py
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
    db_uri = os.getenv(
        'DB_URI',
        'mysql+pymysql://root:dugout2025!!@dugout-dev.cn6mm486utfi.ap-northeast-2.rds.amazonaws.com:3306/dugoutDB?charset=utf8'
    )

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

def save_hitter_data(hitter_df):
    """타자 데이터를 DB에 저장 (기존 데이터 삭제 후)"""
    try:
        engine = get_db_engine()
        table_name = 'hitter_data'

        # 1. 기존 데이터 삭제
        with engine.connect() as connection:
            connection.execute(text(f"TRUNCATE TABLE {table_name};"))
            connection.commit()
        print(f"✅ 기존 '{table_name}' 테이블의 모든 데이터가 삭제되었습니다.")

        # 2. 새로운 데이터 저장
        hitter_df.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"✅ 타자 데이터 {len(hitter_df)}건이 '{table_name}' 테이블에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 타자 데이터 DB 저장 중 오류 발생: {e}")

def save_pitcher_data(pitcher_df):
    """투수 데이터를 DB에 저장 (기존 데이터 삭제 후)"""
    try:
        engine = get_db_engine()
        table_name = 'pitcher_data'

        # 1. 기존 데이터 삭제
        with engine.connect() as connection:
            connection.execute(text(f"TRUNCATE TABLE {table_name};"))
            connection.commit()
        print(f"✅ 기존 '{table_name}' 테이블의 모든 데이터가 삭제되었습니다.")

        # 2. 새로운 데이터 저장
        pitcher_df.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"✅ 투수 데이터 {len(pitcher_df)}건이 '{table_name}' 테이블에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 투수 데이터 DB 저장 중 오류 발생: {e}")

def save_win_probabilities(win_probability_df):
    """승률 예측 DataFrame을 DB에 저장 (기존 데이터 삭제 후)"""
    try:
        engine = get_db_engine()

        # 1. 기존 데이터 삭제
        table_name = 'win_probabilities'
        with engine.connect() as connection:
            connection.execute(text(f"TRUNCATE TABLE {table_name};"))
            connection.commit()
        print(f"✅ 기존 '{table_name}' 테이블의 모든 데이터가 삭제되었습니다.")

        # DataFrame을 DB에 저장하기 좋은 긴 형식(long format)으로 변환
        # 인덱스(team1)를 리셋하여 컬럼으로 만듭니다.
        long_format_df = win_probability_df.reset_index().melt(
            id_vars=['index'], var_name='team2', value_name='win_probability'
        )
        long_format_df = long_format_df.rename(columns={'index': 'team1'})

        # 데이터 타입 변환 및 예측 날짜 추가
        long_format_df['win_probability'] = pd.to_numeric(long_format_df['win_probability'], errors='coerce') # '-' 값 처리
        long_format_df['prediction_date'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 2. 새로운 데이터 저장
        long_format_df.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"✅ 승률 예측 결과 {len(long_format_df)}건이 '{table_name}' 테이블에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 승률 예측 결과 DB 저장 중 오류 발생: {e}")

def save_team_rankings(team_rankings_df):
    """팀 순위 예측 DataFrame을 DB에 저장 (기존 데이터 삭제 후)"""
    try:
        engine = get_db_engine()

        # 1. 기존 데이터 삭제
        table_name = 'team_rankings'
        with engine.connect() as connection:
            connection.execute(text(f"TRUNCATE TABLE {table_name};"))
            connection.commit()
        print(f"✅ 기존 '{table_name}' 테이블의 모든 데이터가 삭제되었습니다.")

        # 예측 날짜 추가
        team_rankings_df['prediction_date'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 2. 새로운 데이터 저장
        team_rankings_df.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"✅ 팀 순위 예측 결과 {len(team_rankings_df)}건이 '{table_name}' 테이블에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 팀 순위 예측 결과 DB 저장 중 오류 발생: {e}")
