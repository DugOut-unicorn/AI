from flask import Flask, request, jsonify
import crawler
import data_processor
import predictor

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False 
# 전역 변수로 데이터와 모델 저장
cached_data = {
    'hitter_data': None,
    'pitcher_data': None,
    'win_probability_df': None,
    'last_update': None
}


@app.route('/')
def home():
    return "KBO 야구 승률 예측 API. '/predict_win_rate' 엔드포인트를 사용하세요."


@app.route('/predict_win_rate', methods=['POST'])
def predict_win_rate():
    # 요청에서 팀 이름 추출
    data = request.get_json()
    if not data or 'team1' not in data or 'team2' not in data:
        return jsonify({'error': '두 팀 이름을 제공해야 합니다. 예: {"team1": "LG", "team2": "삼성"}'}), 400

    team1 = data['team1']
    team2 = data['team2']

    # 데이터 크롤링 및 처리
    try:
        win_probability_df = predictor.get_win_probability_df(cached_data)

        # 유효한 팀 목록 확인
        valid_teams = win_probability_df.index.tolist()

        if team1 not in valid_teams:
            return jsonify({'error': f"'{team1}'은(는) 유효한 팀 이름이 아닙니다. 유효한 팀 목록: {', '.join(valid_teams)}"}), 400
        if team2 not in valid_teams:
            return jsonify({'error': f"'{team2}'은(는) 유효한 팀 이름이 아닙니다. 유효한 팀 목록: {', '.join(valid_teams)}"}), 400
        if team1 == team2:
            return jsonify({'error': "같은 팀 간의 승률은 계산할 수 없습니다."}), 400

        # 승률 계산
        win_prob = win_probability_df.loc[team1, team2]
        if win_prob == '-':
            return jsonify({'error': "승률을 계산할 수 없습니다."}), 400

        # 결과 반환
        return jsonify({
            'team1': team1,
            'team2': team2,
            'win_probability': float(win_prob),
            'message': f"{team1}이(가) {team2}을(를) 상대로 승리할 예측 승률은 {win_prob}% 입니다."
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
