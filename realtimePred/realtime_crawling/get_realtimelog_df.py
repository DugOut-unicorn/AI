import pandas as pd
from .get_log import get_inning_log
from .parsing import parse_log

# inning = 1
# game_id = "20250525NCOB0"
def get_realtimelog_df(inning, game_id) -> pd.DataFrame:
    text_log, away_score, home_score = get_inning_log(game_id, inning)
    score_diff = home_score - away_score

    df_rows = []
    split_data = text_log[0].split('---------------------------------------')

    home_away = 0 # home : 0, away: 1
    for i, part in enumerate(split_data):
        if i ==1: home_away = 1
        if i == 2: break
        
        df_result =  parse_log(part, inning=inning, is_home=home_away)
        ordered_columns = [
            'inning', 'home_away',
            'res_2루타','res_3루타','res_기타','res_땅볼아웃','res_뜬공아웃',
            'res_병살','res_볼넷사구','res_삼진','res_실책',
            'res_안타','res_직선타아웃','res_파울','res_홈런','res_희생'
        ]
        df_result = df_result.reindex(columns=ordered_columns, fill_value=0)
        df_rows.append(df_result)

    df_total = pd.concat(df_rows, ignore_index=True)
    df_total['away_score'] = away_score
    df_total['home_score'] = home_score
    df_total['score_diff'] = score_diff

    df_total = df_total[[
        'inning', 'away_score', 'home_score', 'score_diff', 'home_away',
        'res_2루타','res_3루타','res_기타','res_땅볼아웃','res_뜬공아웃',
        'res_병살','res_볼넷사구','res_삼진','res_실책','res_안타',
        'res_직선타아웃','res_파울','res_홈런','res_희생'
    ]]
    
    return df_total