FROM public.ecr.aws/lambda/python:3.8

RUN pip install opencv-python-headless onnxruntime numpy --target ${LAMBDA_TASK_ROOT}

COPY app.py inference.py cartoonize.onnx deeplabv3.onnx ${LAMBDA_TASK_ROOT}/
CMD ["app.lambda_handler"]