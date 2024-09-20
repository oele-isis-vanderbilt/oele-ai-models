import logging

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, WorkerType, cli



from livekit.rtc import VideoStream, VideoFrame, VideoBufferType
from typing import Tuple
import numpy as np
import cv2

import grpc

from mlserver_jpeg_codec import JPEGCodec
import asyncio

import mlserver.grpc.converters as converters
import mlserver.types as types

import mlserver.grpc.dataplane_pb2_grpc as dataplane
from mlserver.grpc.converters import ModelInferResponseConverter

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)



class MLServerInferer:
    def __init__(self, grpc_url: str, model_name: str, resize_to: Tuple[int, int]) -> None:
        self.grpc_url = grpc_url
        self.model_name = model_name
        self.resize_to = resize_to
        self.tasks = {}


    async def infer_from(self, video_stream: VideoStream):
        task = InferenceTask(self.model_name, self.grpc_url, video_stream, self.resize_to)
        aio_task = asyncio.ensure_future(task.start())
        self.tasks[video_stream._track.sid] = task
        return task    
    
    def stop(self, track_sid):
        task = self.tasks.get(track_sid)
        if task:
            task.stop()
            del self.tasks[track_sid]


class InferenceTask:
    def __init__(self, model_name: str, grpc_url: str, video_stream: VideoStream, resize_to: Tuple[int, int]) -> None:
        self.model_name = model_name
        self.grpc_url = grpc_url
        self.video_stream = video_stream
        self.inference_request = None
        self.resize_to = resize_to
        self.frame_queue = asyncio.Queue(maxsize=1)
        self.stop_event = asyncio.Event()
        self.stop_event.clear()

    async def start(self):
        async with grpc.aio.insecure_channel("localhost:8081", options=[]) as grpc_channel:
            grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)
            async for response in grpc_stub.ModelStreamInfer(self._create_inference_request()):
                response = ModelInferResponseConverter.to_types(response)
                arr = JPEGCodec.decode_output(response.outputs[0]) 
                width, height = arr.shape[1], arr.shape[0]

                frame = VideoFrame(
                    width=width,
                    height=height,
                    type=VideoBufferType.RGB24,
                    data=arr.tobytes(),
                )
                # logger.info("Updated frame result")
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()  # Remove the old frame if the queue is full
                self.frame_queue.put_nowait(frame)
                if self.stop_event.is_set():
                    break

    async def __anext__(self):
        frame_result = await self.frame_queue.get()
        # logger.info("Waiting for frame result")
        return frame_result
    
    def __aiter__(self):
        return self
    
    def stop(self):
        self.stop_event.set()

    async def _create_inference_request(self):
        count = 0
        async for frame_event in self.video_stream:
            count += 1
            # if idx % 2 == 0:
            #     continue
            if count % 2 == 0:
                continue
            frame = frame_event.frame
            try:
                # resize frame
                arr = np.frombuffer(frame.data, dtype=np.uint8).reshape((frame.height, frame.width, 3))
                resized = cv2.resize(arr, self.resize_to, interpolation=cv2.INTER_AREA)
                # encode frame
                encoded_frame = JPEGCodec.encode_input(name="payload", payload=resized, use_bytes=True)
                inference_request_g = converters.ModelInferRequestConverter.from_types(
                        types.InferenceRequest(inputs=[encoded_frame]), 
                        model_name="yolo-jpeg-codec", 
                        model_version=None
                    )
                yield inference_request_g
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                raise e


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    await ctx.connect(auto_subscribe=AutoSubscribe.VIDEO_ONLY)

    room = ctx.room

    logger.info(f"room: {room.name}")

    inferer = MLServerInferer("localhost:8081", "yolo-jpeg-codec", (1280, 720))
    published_track_map = {}

    @room.on("track_subscribed")
    def on_new_track(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.Participant,
    ):
        video_stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
        asyncio.create_task(track_task(video_stream, participant.name or participant.identity))

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.Participant,
    ):
        logger.info(f"Unsubscribed from track {track.sid}")
        if track.sid in inferer.tasks:
            inferer.stop(track.sid)
        if track.sid in published_track_map:
            asyncio.create_task(room.local_participant.unpublish_track(published_track_map[track.sid]))

    
    async def track_task(video_stream, id):
        source = rtc.VideoSource(1280, 720)
        track_published = rtc.LocalVideoTrack.create_video_track(
            source=source, name=f"{id}-yolo"
        )
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA
        )
        publication = await room.local_participant.publish_track(track_published, options)
        published_track_map[video_stream._track.sid] = publication.track.sid
        published = await inferer.infer_from(video_stream)
        logger.info("starting inference for track {}")
        try:
            async for frame in published:
                source.capture_frame(frame)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise e
        


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM, port=9000))