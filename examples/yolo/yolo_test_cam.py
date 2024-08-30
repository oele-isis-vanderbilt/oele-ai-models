import cv2

import logging
from mlserver.codecs import NumpyCodec
from mlserver.types import InferenceRequest
import requests
inference_url = "http://localhost:8080/v2/models/yolo/infer"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    device = cv2.VideoCapture(2)

    # Set 30 FPS
    device.set(cv2.CAP_PROP_FPS, 30)
    # Set 1280x720 resolution
    device.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    device.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logging.info("Starting camera...")
    count = 0
    while True:
        ret, frame = device.read()
        if not ret:
            break
        
        logging.info("Sending request...")

        inference_request = InferenceRequest(
        inputs=[
            NumpyCodec.encode_input(name="payload", payload=frame)
        ]
        )

        res = requests.post(inference_url, json=inference_request.dict())
        res.raise_for_status()  # Check for HTTP request errors
        logging.info("Got Response...")
        response_dict = res.json()
        print(response_dict)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    device.release()
    