import requests
import freddy
from app import ModelInput

BASE_URL = "http://0.0.0.0:8000"
PREDICT_URL = BASE_URL + "/predict"

if __name__ == "__main__":
    print(ModelInput.schema())

    for i in range(10):
        request_data = [
            freddy.sample(ModelInput)
        ]
        print(request_data)
        response = requests.post(
            PREDICT_URL,
            json=request_data,
        )
        print(response.status_code)
        print(response.json())