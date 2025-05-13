FROM python:3.10-bookworm

WORKDIR /app

# requirements 먼저 복사하고 설치
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
