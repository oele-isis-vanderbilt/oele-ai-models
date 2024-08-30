import grpc
import mlserver.types as types
from mlserver.grpc.converters import ModelInferResponseConverter
import mlserver.grpc.converters as converters
import mlserver.grpc.dataplane_pb2_grpc as dataplane
from mlserver.codecs import NumpyCodec
import cv2
# inference_request = types.InferenceRequest.parse_file("./generate-request.json")

# need to convert from string to bytes for grpc
# inference_request.inputs[0] = StringCodec.encode_input("prompt", inference_request.inputs[0].data.root)
# inference_request_g = converters.ModelInferRequestConverter.from_types(
#     inference_request, model_name="text-model", model_version=None
# )

device = cv2.VideoCapture(2)

# Set 30 FPS
device.set(cv2.CAP_PROP_FPS, 30)
# Set 1280x720 resolution
device.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
device.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


async def image_stream_generator():
    while True:
        # Encode the frame as needed (e.g., convert to bytes)
        _, frame = device.read()
        encoded_frame = NumpyCodec.encode_input(name="payload", payload=frame)
        
        # Create the gRPC inference request
        inference_request_g = converters.ModelInferRequestConverter.from_types(
            types.InferenceRequest(inputs=[encoded_frame]), 
            model_name="yolo", 
            model_version=None
        )
        yield inference_request_g

    

async def main():
    import json
    response_queue = asyncio.Queue()

    async def response_consumer(q):
        while True:
            response = await q.get()
            arr = NumpyCodec.decode_output(response.outputs[0])
            cv2.imshow("Frame", arr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    asyncio.create_task(response_consumer(response_queue))
    async with grpc.aio.insecure_channel("localhost:8081", options=[('grpc.max_send_message_length', -1),
                 ('grpc.max_receive_message_length', -1)]) as grpc_channel:
        grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)
        async for response in grpc_stub.ModelStreamInfer(image_stream_generator()):
            response = ModelInferResponseConverter.to_types(response)
            response_queue.put_nowait(response)        
            


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    device.release()
