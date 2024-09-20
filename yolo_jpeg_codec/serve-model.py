from typing import AsyncIterator
import numpy as np
from ultralytics import YOLO
from mlserver import types
from mlserver.model import MLModel
# from mlserver.codecs import NumpyCodec
from mlserver_jpeg_codec import JPEGCodec
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YoloModel(MLModel):

    def __init__(self, settings):
        super().__init__(settings)
        self.predict_stream = self.predict_stream

    async def load(self) -> bool:
        try:
            logger.info("Loading model...")
            self.model = YOLO("yolov8n.pt")
            self.model.to('cuda')
            self.verbose = False
            logger.info("Model loaded successfully.")
            logger.info(f"Model Device: {self.model.device}")
            self.ready=True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.ready = False
        return self.ready

    def _preprocess_inputs(self, payload: types.InferenceRequest):
        frame = []
        for inp in payload.inputs:
            # frame = self.decode(inp, default_codec=NumpyCodec)
            frame = JPEGCodec.decode_input(inp)
        logger.debug("Inputs preprocessed.")
        return frame

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        try:
            # logger.info("Processing prediction request...")
            bgr_frame = self._preprocess_inputs(payload)

            results = self.model.predict(bgr_frame, verbose=self.verbose, device='cuda:0', classes=[0])

            op_frame = results[0].plot()
            
            return types.InferenceResponse(
                id=payload.id,
                model_name=self.name,
                model_version=self.version,
                outputs=[
                    JPEGCodec.encode_output(name="output", payload=op_frame)
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
    
    async def predict_stream(self, payloads: AsyncIterator[types.InferenceRequest]) -> AsyncIterator[types.InferenceResponse]:
        async for payload in payloads:
            yield await self.predict(payload)
    
        