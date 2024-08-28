import torch
import numpy as np
from ultralytics import YOLO
from mlserver import types
from mlserver.model import MLModel
from mlserver.codecs import NumpyCodec
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YoloModel(MLModel):
    async def load(self) -> bool:
        try:
            logger.info("Loading model...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = YOLO("yolov8n.pt")
            self.verbose = True
            logger.info("Model loaded successfully.")
            logger.info(f"Torch device: {self.device}")
            logger.info(f"Yolo Model device: {self.model.device}")
            self.ready=True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.ready = False
        return self.ready

    def _preprocess_inputs(self, payload: types.InferenceRequest):
        frame = []
        for inp in payload.inputs:
            frame = self.decode(inp, default_codec=NumpyCodec)
        logger.debug("Inputs preprocessed.")
        return frame

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        try:
            logger.info("Processing prediction request...")
            bgr_frame = self._preprocess_inputs(payload)

            results = self.model.predict(bgr_frame, verbose=self.verbose)
            annotated_frame = results[0].plot()

            logger.debug("Prediction processed successfully.")

            return types.InferenceResponse(
                id=payload.id,
                model_name=self.name,
                model_version=self.version,
                outputs=[
                    types.ResponseOutput(
                        name="frame",
                        shape=[len(annotated_frame)],
                        datatype="BYTES",
                        data=np.asarray(annotated_frame).tolist(),
                    )
                ],
            )

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Return an empty response or handle the error as needed
            return types.InferenceResponse(
                id=payload.id,
                model_name=self.name,
                model_version=self.version,
                outputs=[]
            )