FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .
ENV FLASK_APP=main.py
CMD ["flask", "run", "--host=0.0.0.0"]