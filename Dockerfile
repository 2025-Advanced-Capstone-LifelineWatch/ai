FROM python:3.10-slim

WORKDIR /app

# GPG 키 수동 등록
RUN apt-get update || true && apt-get install -y gnupg curl

# 필요한 GPG 키 수동 추가
RUN gpg --keyserver keyserver.ubuntu.com --recv-keys 0E98404D386FA1D9 \
    && gpg --keyserver keyserver.ubuntu.com --recv-keys 6ED0E7B82643E131 \
    && gpg --keyserver keyserver.ubuntu.com --recv-keys 54404762BBB6E853 \
    && gpg --keyserver keyserver.ubuntu.com --recv-keys BDE6D2B9216EC7A8 \
    && gpg --keyserver keyserver.ubuntu.com --recv-keys F8D2585B8783D481

# 키를 APT에 등록
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0E98404D386FA1D9 6ED0E7B82643E131 54404762BBB6E853 BDE6D2B9216EC7A8 F8D2585B8783D481

# 패키지 설치
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]
