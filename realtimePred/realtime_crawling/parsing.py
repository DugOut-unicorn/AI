from collections import defaultdict
import pandas as pd
import re

# 카테고리 리스트
category_keywords = {
    'res_안타': ['안타', '1루타'],
    'res_2루타': ['2루타'],
    'res_3루타': ['3루타'],
    'res_홈런': ['홈런'],
    'res_땅볼아웃': ['땅볼'],
    'res_직선타아웃': ['직선타'],
    'res_뜬공아웃': ['플라이 아웃', '뜬공', '플라이', '플라이아웃', '파울플라이'],
    'res_병살': ['병살'],
    'res_희생': ['희생플라이', '희생번트'],
    'res_볼넷사구': ['볼넷', '4구', '사구', '고의4구', '고4', '몸에 맞는 볼'],
    'res_삼진': ['삼진'],
    'res_실책': ['실책'],
    'res_기타': ['피치클락', '투수 플라이']
}

# 결과 추출 함수
def get_result_from_line(line) -> None:
    for category, keywords in category_keywords.items():
        for kw in keywords:
            if kw in line:
                return category
    return None

# 로그 파싱 함수
def parse_log(log_text, inning, is_home) -> pd.DataFrame:
    blocks = re.split(r'\n\s*\n', log_text.strip())
    event_counter = defaultdict(int)

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        for line in lines:
            line = line.strip()
            # 이름 : 텍스트만
            if re.match(r'^\d루주자', line):
                continue
            if re.match(r'^[가-힣]+ ?[가-힣]* ?:', line):
                result = get_result_from_line(line)
                if result:
                    event_counter[result] += 1
                break 

    row = {'inning': inning, 'home_away': is_home}
    for key in category_keywords:
        row[key] = event_counter.get(key, 0)

    return pd.DataFrame([row])
