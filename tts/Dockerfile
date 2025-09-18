FROM python:3.12-slim

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 1375

CMD ["python", "app.py"]