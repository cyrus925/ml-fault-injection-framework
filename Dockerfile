FROM python:3.14.2

WORKDIR /fault-injection-app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python","./orchestator.py"]