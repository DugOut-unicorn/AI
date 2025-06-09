from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time

# 배포할 때 쓰라신다

def get_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage') 
    options.add_argument('--window-size=1280x720') 
    # chrome_service = ChromeService(executable_path="/usr/lib/chromium/chromedriver")
    return webdriver.Chrome(options=options)

def get_inning_log(game_id: str, inning_index: int) -> list[list, int, int]:
    # 크롬열기
    driver =  get_driver() # webdriver.Chrome()
    url = f"https://www.koreabaseball.com/Game/LiveText.aspx?leagueId=1&seriesId=0&gameId={game_id}0&gyear=2025"
    driver.get(url)
    wait = WebDriverWait(driver, 10)
    
    # 누적 스코어 계산
    wait.until(EC.presence_of_element_located((By.ID, "tblScoreBoard2")))

    def sumScore(team):
        total_score = 0
        for i in range(1, inning_index + 1):
            try:
                td = driver.find_element(By.ID, f"rptScoreBoard2_tdInn{i}_{team}")
                text = td.text.strip()
                if text.isdigit():
                    total_score += int(text)
            except:
                pass
        return total_score
    
    away_total = sumScore(0)
    home_total = sumScore(1)
    print(f"{inning_index}회까지 어웨이 누적점수: {away_total}")
    print(f"{inning_index}회까지 홈 누적점수: {home_total}")
    
    # 이닝별 로그 추출
    tab_xpath = f'//*[@id="numNav"]/li[{inning_index}]'
    tab_btn = wait.until(EC.element_to_be_clickable((By.XPATH, tab_xpath)))
    tab_btn.click()
    log_ul_xpath = f'//*[@id="numCont{inning_index}"]'
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, log_ul_xpath)))
        logs = driver.find_elements(By.XPATH, log_ul_xpath)
        log_texts = [log.text for log in logs]
        
        # 로그와 누적 스코어 리턴
        return log_texts, away_total, home_total
    except:
        print(f"\n{inning_index}회 로그를 찾을 수 없습니다.")

    driver.quit()