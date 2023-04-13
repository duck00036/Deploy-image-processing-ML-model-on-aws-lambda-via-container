# Deploy image processing ML model on aws lambda via container
This repository provide a detailed guide on how to deploy a machine learning (ML) model for image processing on AWS Lambda using a container. The step-by-step process will be presented, making it easier for users to follow and replicate the deployment on their own.

## Why aws lambda?
AWS Lambda is a serverless computing platform, which means that you don't need to provision, manage, or scale any infrastructure themselves. Instead, AWS Lambda automatically runs code in response to events, which can be triggered by various services within the AWS ecosystem. This makes it a highly scalable and cost-effective solution for running short-lived compute tasks, such as processing images with ML models.

## Why container
Using containers to deploy ML models on AWS Lambda can simplify the deployment process, improve isolation and security, increase portability, and provide greater control over the runtime environment. By packaging all the dependencies and runtime libraries needed to run the ML model in a self-contained container, you can easily deploy and maintain the application while ensuring consistency across environments.

# Architecture
In this repository, I will use the AWS s3 bucket put event as a trigger and another s3 bucket to hold the output file, so the architecture will look like this:
![d6](https://user-images.githubusercontent.com/48171500/231827076-b32edbb4-1656-48be-b777-7935635a1870.PNG)

# Prerequisite
Before deployment, make sure you have:
* an environment with python3 and docker
* a trained ML model

# Step 1 : Create S3 buckets
## though AWS CLI
AWS CLI should be installed on the system.

or you can install it using following commands:
```
sudo apt udpate
sudo apt install awscli
```
and check if the aws cli is installed correctly by following command:
```
aws --version
```
Now you can use your **AWS Access Key ID** and **AWS Secret Access Key** to configure AWS CLI:
```
aws configure
```
![d7](https://user-images.githubusercontent.com/48171500/231861076-2d656547-143b-4292-b72b-c30a801067e6.PNG)

Before creating, you can first check the S3 buckets you already have by following command:
```
aws s3 ls
```
Now, using following commands to creat your own input and ouput buckets:
```
aws s3 mb s3://<your-input-bucket-name> --region <your-region>
aws s3 mb s3://<your-output-bucket-name> --region <your-region>
```
If buckets are created successfully, they will appear in the s3 list.

## though AWS console
Open the console page and create your own input and output buckets, it's easy.

# Step 2 : Write application code
You will need to have an application code that you want to containerize.

In the applicatioin code, we should first import some necessary dependencies:
```py
# necessary dependencies
import os
import boto3
s3 = boto3.client('s3')

# your model's dependencies
import onnxruntime as rt
import cv2
import inference 
```

Then we need to define a lambda handler function to handle the event, when your function is called, Lambda will run the handler function.
```py
def lambda_handler(event, context):
```
In the function, first get the name of the trigger bucket and the name of the image object.
```py
    # Get the bucket and key of the uploaded image
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
```
Second, download the image from the trigger bucket through boto3 and get image name from the name of image object.
```py    
    # Download the image from S3
    local_filename = '/tmp/image.jpg'
    s3.download_file(bucket_name, key, local_filename)
    image_name = os.path.splitext(key)[0]
```
Then, you can load your ML model and do image processing.

(This block is just an example, please replace it with your own code)
```py 
    # Load deeplab and cartoonize model though onnx
    deeplab = rt.InferenceSession('deeplabv3.onnx',providers=['CPUExecutionProvider'])
    cartoon = rt.InferenceSession('cartoonize.onnx',providers=['CPUExecutionProvider'])
    
    # Import image
    image = cv2.imread(local_filename)
    image = inference.resize_crop(image)
    
    # Predict person masks 
    mask1,mask2 = inference.findmask(image, deeplab)
      
    # Transfer image to cartoon style
    cartoon_img = inference.cartoonize(image, cartoon)
    
    # Compose mask and cartoonized image into final result
    person = cv2.bitwise_and(cartoon_img, cartoon_img, mask=mask1)
    background = cv2.bitwise_and(image, image, mask=mask2)
    output = person + background
    
    # Save result image
    output_filename = '/tmp/output.jpg'
    cv2.imwrite(output_filename, output)
```
After image processing, upload the output image to the output s3 bucket .
```py    
    # Upload the results to another S3 bucket
    result_bucket_name = os.getenv('OUTPUT_BUCKET', None)
    image_result_key = image_name + '.jpg'
    s3.upload_file(output_filename, result_bucket_name, image_result_key)
```
(Environment variables will be defined later in the lambda function)

# Step 3 : Write the Dockerfile
A Dockerfile is a text file that contains a set of instructions for building a Docker image. You will need to create a Dockerfile that includes instructions for installing any dependencies required by your application, copying the application code into the image, and setting up the environment.

AWS provided base images for Lambda contain all the required components to run your functions packaged as container images on AWS Lambda.

We can use it as the base image:
```Dockerfile
FROM public.ecr.aws/lambda/python:3.8
```
Then, install the dependencies required by your model and code
```Dockerfile
RUN pip install opencv-python-headless onnxruntime numpy --target ${LAMBDA_TASK_ROOT}
```
Copy your model and code into the working directory
```Dockerfile
COPY app.py inference.py cartoonize.onnx deeplabv3.onnx ${LAMBDA_TASK_ROOT}/
```
Last, add a command to run the lambda handler when the lambda function is triggered
```Dockerfile
CMD ["app.lambda_handler"]
```

# Step 4 : Build a container and push to ECR



