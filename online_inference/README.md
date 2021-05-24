FastAPI сервис для онлайн-инференса ML-модели классификации болезней сердца 

Запуск локально (предполагается наличие sklearn piсkle модели в файле, путь к которому прописан в переменной окружения $PATH_TO_MODEL
): 
~~~
cd online_inference
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
python make_request.py
pytest
~~~
Сборка и запуск docker-образа:
~~~
cp $PATH_TO_MODEL model.pkl
docker build -t ml_in_prod/online_inference:v1 .
docker run -p 8000:80 ml_in_prod/online_inference:v1
~~~
Загрузка образа с dockerhub-a:
~~~
docker pull esoldatenko/online_inference:v1 
docker run -p 8000:80 esoldatenko/online_inference:v1
~~~