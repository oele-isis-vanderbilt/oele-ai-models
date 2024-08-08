import logging
from mlserver.codecs import NumpyCodec
from mlserver.types import InferenceRequest
import requests
import numpy as np
import sys
import os
import cv2  # To handle the image encoding and decoding

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Inference variables
try:
    if len(sys.argv) < 2:
        logging.error("Please provide inference mode (--local or --remote)")
        sys.exit(1)

    if sys.argv[1] == "--local":
        inference_url = 'http://0.0.0.0:8080/v2/models/emonet/infer'
    elif sys.argv[1] == "--remote":
        inference_url = 'http://localhost:8080/seldon/default/hsemotion/v2/models/infer'
    else:
        logging.error("Invalid inference mode provided. Use --local or --remote.")
        sys.exit(1)

except Exception as e:
    logging.error(f"Error setting inference URL: {e}")
    sys.exit(1)

# Load and preprocess the image
try:
    logging.info("Loading and preprocessing image...")
    image_path = "./images/multiple-faces-emotions.jpg"
    input_data = cv2.imread(image_path)
    print(input_data)

    if input_data is None:
        logging.error(f"Failed to load image from path: {image_path}")
        sys.exit(1)

except Exception as e:
    logging.error(f"Error loading and preprocessing image: {e}")
    sys.exit(1)

# Build the inference request
try:
    inference_request = InferenceRequest(
        inputs=[
            NumpyCodec.encode_input(name="payload", payload=input_data)
        ]
    )
except Exception as e:
    logging.error(f"Error building inference request: {e}")
    sys.exit(1)

# Send the inference request and capture response
try:
    logging.info("Sending Inference Request...")
    res = requests.post(inference_url, json=inference_request.dict())
    res.raise_for_status()  # Check for HTTP request errors
    logging.info("Got Response...")

    response_dict = res.json()

except requests.RequestException as e:
    logging.error(f"Error sending inference request or receiving response: {e}")
    sys.exit(1)
except ValueError as e:
    logging.error(f"Error parsing JSON response: {e}")
    sys.exit(1)

# Parse and process the response
try:
    output_array = res.json()
    image_array = np.asarray(output_array['outputs'][0]['data'])
    cv2.imwrite('multiple-faces-emotions-output.png',image_array)
    logging.info("Image saved as output_image.png")

except KeyError as e:
    logging.error(f"Key error in response data: {e}")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error processing response data: {e}")
    sys.exit(1)
