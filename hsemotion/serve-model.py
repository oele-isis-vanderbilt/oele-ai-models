import torch
import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from facenet_pytorch import MTCNN
from mlserver import types
from mlserver.model import MLModel
from mlserver.codecs import NumpyCodec
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HSEmotionModel(MLModel):
    async def load(self) -> bool:
        try:
            logger.info("Loading model...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.face_detector = MTCNN()
            self.emotion_detector = HSEmotionRecognizer(
                model_name='enet_b0_8_va_mtl')
            self.ready = True
            logger.info("Model loaded successfully.")
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

            faceboxes, conf, landmarks = self.face_detector.detect(
                bgr_frame, landmarks=True)

            if faceboxes is not None:
                faceboxes = faceboxes[conf > 0.9]
                landmarks = landmarks[conf > 0.9]

            if landmarks is not None:
                for landmark in landmarks:
                    for (x, y) in landmark.astype(int):
                        cv2.circle(bgr_frame, (x, y), 2, (255, 0, 0), -1)

            if faceboxes is not None:
                for box in faceboxes:
                    box = box.astype(int)
                    cv2.rectangle(
                        bgr_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    face_crop = bgr_frame[box[1]:box[3], box[0]:box[2]]
                    emotion, values = self.emotion_detector.predict_emotions(
                        face_crop)

                    logger.info(f"Detected emotion: {emotion}, values: {values}")

                    valence = values[-2]
                    arousal = values[-1]

                    text_emotion = f"Emo: {emotion}"
                    text_valence = f"V:  {valence:.2f}"
                    text_arousal = f"A: {arousal:.2f}"

                    cv2.putText(bgr_frame, text_emotion, (box[0] + 10, box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    bbox_left_center = (box[0], (box[1] + box[3]) // 2)
                    bbox_right_center = (box[2], (box[1] + box[3]) // 2)

                    cv2.putText(bgr_frame, text_valence,
                                (bbox_left_center[0] - 80, bbox_left_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(bgr_frame, text_arousal,
                                (bbox_right_center[0] + 10, bbox_right_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            logger.debug("Prediction processed successfully.")

            return types.InferenceResponse(
                id=payload.id,
                model_name=self.name,
                model_version=self.version,
                outputs=[
                    types.ResponseOutput(
                        name="frame",
                        shape=[len(bgr_frame)],
                        datatype="BYTES",
                        data=np.asarray(bgr_frame).tolist(),
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

    def _get_emotion_text(self, bgr_frame: np.ndarray) -> str:
        try:
            logger.info("Getting emotion text...")
            faceboxes, conf = self.face_detector.detect(
                bgr_frame, landmarks=False)

            if faceboxes is not None:
                faceboxes = faceboxes[conf > 0.9]

            if faceboxes is not None:
                for box in faceboxes:
                    box = box.astype(int)
                    face_crop = bgr_frame[box[1]:box[3], box[0]:box[2]]
                    emotion, values = self.emotion_detector.predict_emotions(
                        face_crop)
                    valance = values[-2]
                    arousal = values[-1]
                    emotion_text = f"Emotion: {emotion} Valence: {valance:.2f} Arousal: {arousal:.2f}"
                    logger.info(f"Emotion text: {emotion_text}")
                    return emotion_text
        except Exception as e:
            logger.error(f"Error getting emotion text: {e}")

        return "UNKNOWN"
