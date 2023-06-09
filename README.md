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
configure process will be like this:
```
AWS Access Key ID [None]:your AWS Access Key ID
AWS Secret Access Key [None]:your AWS Secret Access Key
Default region name [None]:your region name
Default output format [None]:json
```

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

### Notice : You can also use AWS console to create your own input and output buckets.

# Step 2 : Write application code
You will need to have an application code that you want to containerize.

In the applicatioin code, we should first import some necessary dependencies.
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
After image processing, upload the output image to the output s3 bucket.
```py    
    # Upload the results to another S3 bucket
    result_bucket_name = os.getenv('OUTPUT_BUCKET', None)
    image_result_key = image_name + '.jpg'
    s3.upload_file(output_filename, result_bucket_name, image_result_key)
```
Environment variables will be defined later in the lambda function.

# Step 3 : Write the Dockerfile
A Dockerfile is a text file that contains a set of instructions for building a Docker image. You will need to create a Dockerfile that includes instructions for installing any dependencies required by your application, copying the application code into the image, and setting up the environment.

First, create a Dockerfile:
```
touch Dockerfile
```

AWS provided base images for Lambda contain all the required components to run your functions packaged as container images on AWS Lambda.

We can use it as the base image in our Dockerfile
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
First, build your docker image with the docker build command:
```
docker build -t <image-name>:<tag> .
```
Run the get-login-password command to authenticate the Docker CLI to your Amazon ECR registry:
```
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-AWS-account-ID>.dkr.ecr.<your-region>.amazonaws.com
```
Create a repository in Amazon ECR using the create-repository command:
```
aws ecr create-repository --repository-name <image-name> --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE
```
If successful, you will see a response like this:
```JSON
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:<your-region>:<your-AWS-account-ID>:repository/<image-name>",
        "registryId": "<your-AWS-account-ID>",
        "repositoryName": "<image-name>",
        "repositoryUri": "<your-AWS-account-ID>.dkr.ecr.<your-region>.amazonaws.com/<image-name>",
        "createdAt": "2023-03-09T10:39:01+00:00",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": true
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}
```

### Notice : You can also use AWS console to creat repository in Amazon ECR.

After creating a repository in Amazon ECR, we need to tag our docker image and push to Amazon ECR repository.

Tag our docker image using docker tag command:
```
docker tag <image-name>:<tag> <your-AWS-account-ID>.dkr.ecr.<your-region>.amazonaws.com/<image-name>:<tag>
```
Then run the docker push command to deploy your local image to the Amazon ECR repository:
```
docker push <your-AWS-account-ID>.dkr.ecr.<your-region>.amazonaws.com/<image-name>:<tag>
```
Now, we have our docker image in Amazon ECR repository.

# Step 5 : Create a AWS Lambda function using container
Before creating lambda funcion, we need to creat a policy and an IAM role first.

First, we should create a trust-policy JSON file for creating IAM role:
```
touch trust-policy.json
```
json file should contain the following:
```JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```
Now, we can create a IAM role for lambda function by following command:
```
aws iam create-role --role-name lambda-s3-role --assume-role-policy-document file://./trust-policy.json
```
If successful, you will see a response like this:
```JSON
{
    "Role": {
        "Path": "/",
        "RoleName": "lambda-s3-role",
        "RoleId": "AROAVSEX6VQLZ3GBRHLRH",
        "Arn": "arn:aws:iam::<your-AWS-account-ID>:role/lambda-s3-role",
        "CreateDate": "2023-04-14T14:52:04+00:00",
        "AssumeRolePolicyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
    }
}
```

Then we need to create a JSON file to put the policy into the IAM role:
```
touch role-policy.json
```
json file should contain the following:
```JSON
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:PutLogEvents",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": "*"
        }
    ]
}
```
We can now apply the policy to the IAM role we just created by following command:
```
aws iam put-role-policy --role-name lambda-s3-role --policy-name lambda-s3-policy --policy-document file://./role-policy.json
```
So far we have an IAM role with s3 and cloudwatch access.

### Notice : You can also use AWS console to creat IAM role and policy.

Finally, we can create an AWS Lambda function now.

You can use following command to create a lambda function using container:
```
aws lambda create-function \
--function-name <your-function-name> \
--package-type Image \
--code ImageUri=<your-AWS-account-ID>.dkr.ecr.<your-region>.amazonaws.com/<image-name>:<tag> \
--role arn:aws:iam::<your-AWS-account-ID>:role/lambda-s3-role
```
If successful, you will see a response like this:
```JSON
{
    "FunctionName": "<your-function-name>",
    "FunctionArn": "arn:aws:lambda:<your-region>:<your-AWS-account-ID>:function:<your-function-name>",
    "Role": "arn:aws:iam::<your-AWS-account-ID>:role/lambda-s3-role",
    "CodeSize": 0,
    "Description": "",
    "Timeout": 3,
    "MemorySize": 128,
    "LastModified": "2023-04-14T13:36:13.955+0000",
    "CodeSha256": "e28e2bd18e74c48a3ad08ee9410aaa7ee030c40230ef83627c8a725149527c6e",
    "Version": "$LATEST",
    "TracingConfig": {
        "Mode": "PassThrough"
    },
    "RevisionId": "a5273b7d-1b7b-4159-b753-4833518dfcb3",
    "State": "Pending",
    "StateReason": "The function is being created.",
    "StateReasonCode": "Creating",
    "PackageType": "Image",
    "Architectures": [
        "x86_64"
    ],
    "EphemeralStorage": {
        "Size": 512
    },
    "SnapStart": {
        "ApplyOn": "None",
        "OptimizationStatus": "Off"
    }
}
```
### Notice : You can also use AWS console to creat lambda funcion.

Next, we need to set our output bucket as an environment variable in the lambda function, also, if your model needs larger memory size and running time, you can also update them at the same time by the following command:
```
aws lambda update-function-configuration \
--function-name <your-function-name> \
--environment "Variables={OUTPUT_BUCKET=<your-output-bucket-name>}" \
--memory-size <memory size you need> \
--timeout <running time you need>
```
### Notice : You can also use AWS console to set them.

# Step 6 : Add S3 bucket trigger
Adding a trigger to Lambda function need two step.

First we need to grant permission to the S3 bucket to invoke the Lambda function, this can be done with the following command:
```
aws lambda add-permission \
--function-name <your-function-name> \
--action lambda:InvokeFunction \
--principal s3.amazonaws.com \
--source-arn arn:aws:s3:::<your-input-bucket-name> \
--source-account <your-AWS-account-ID> \
--statement-id s3-trigger
```
We can use the get-policy function to ensure the permission is added:
```
aws lambda get-policy --function-name <your-function-name>
```
After adding a permission, we need to configure S3 notification, and we should first create a JSON file to describes the trigger:
```
touch Notification.json
```
json file should contain the following:
```JSON
{
"LambdaFunctionConfigurations": [
    {
      "Id": "lambda-s3-event-configuration",
      "LambdaFunctionArn": "arn:aws:lambda:<your-region>:<your-AWS-account-ID>:function:<your-function-name>",
      "Events": [ "s3:ObjectCreated:Put" ]
    }
  ]
}
```
As a final step, we configure S3 notifications with the following command:
```
aws s3api put-bucket-notification-configuration --bucket <your-input-bucket-name> --notification-configuration file://./Notification.json
```
### Notice : You can also use AWS console to add S3 bucket trigger.

# Congratulations !

**Finally, we have an image processing ML model on aws lambda via container, triggered by s3 bucket !**

**If no errors occur, put the image in the input bucket and you will get the image processed by your model in the output bucket !**



