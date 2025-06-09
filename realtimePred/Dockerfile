FROM python:3.11-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    wget curl unzip gnupg ca-certificates \
    chromium-driver chromium \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 크롬 환경 설정
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/lib/chromium/chromedriver
ENV PATH=$CHROMEDRIVER_PATH:$PATH

# 작업 디렉토리 설정
WORKDIR /app

# 코드 복사
COPY . .

# 패키지 설치
RUN pip install --upgrade pip && pip install -r requirements.txt

# 실행
CMD ["python", "app.py"]