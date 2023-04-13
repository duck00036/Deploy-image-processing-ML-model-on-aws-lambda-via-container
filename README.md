# Deploy image processing ML model on aws lambda via container
This repository provide a detailed guide on how to deploy a machine learning (ML) model for image processing on AWS Lambda using a container. The step-by-step process will be presented, making it easier for users to follow and replicate the deployment on their own.

## Why aws lambda?
AWS Lambda is a serverless computing platform, which means that you don't need to provision, manage, or scale any infrastructure themselves. Instead, AWS Lambda automatically runs code in response to events, which can be triggered by various services within the AWS ecosystem. This makes it a highly scalable and cost-effective solution for running short-lived compute tasks, such as processing images with ML models.

## Why container
Using containers to deploy ML models on AWS Lambda can simplify the deployment process, improve isolation and security, increase portability, and provide greater control over the runtime environment. By packaging all the dependencies and runtime libraries needed to run the ML model in a self-contained container, you can easily deploy and maintain the application while ensuring consistency across environments.

# Preparation
Before deployment, make sure you have:
* an environment with python3 and docker
* a trained ML model

# Architecture
In this repository, I will use the AWS s3 bucket put event as a trigger and another s3 bucket to hold the output file, so the architecture will look like this:
![d6](https://user-images.githubusercontent.com/48171500/231827076-b32edbb4-1656-48be-b777-7935635a1870.PNG)


