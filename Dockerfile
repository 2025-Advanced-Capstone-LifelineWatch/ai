FROM python:3.10-bookworm

WORKDIR /app

COPY requirements.txt .

# pip 업그레이드 & 기본 도구들 강화
RUN pip install --upgrade pip setuptools wheel

# 메모리 안정성 확보 위해 분리 설치
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]
