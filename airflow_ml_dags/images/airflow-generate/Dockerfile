FROM python:3.9-slim-buster

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /data/
COPY generator.py /generator.py

ENTRYPOINT ["python", "generator.py"]