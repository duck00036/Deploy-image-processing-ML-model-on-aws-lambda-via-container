FROM public.ecr.aws/lambda/python:3.8

RUN yum update
RUN yum install -y tzdata
RUN yum install -y python3-pip git zip curl htop mesa-libGL libglib2.0-0 libpython3-dev gnupg g++
RUN python3 -m pip install --upgrade pip wheel
RUN pip install opencv-python onnxruntime numpy --target ${LAMBDA_TASK_ROOT}

COPY app.py inference.py cartoonize.onnx deeplabv3.onnx ${LAMBDA_TASK_ROOT}/
CMD ["app.lambda_handler"]





