import os
import boto3
import onnxruntime as rt
import cv2
import inference 

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Get the bucket and key of the uploaded image
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Download the image from S3
    local_filename = '/tmp/image.jpg'
    s3.download_file(bucket_name, key, local_filename)
    image_name = os.path.splitext(key)[0]
    
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
    
    # Upload the results to another S3 bucket
    result_bucket_name = os.getenv('OUTPUT_BUCKET', None)
    image_result_key = image_name + '.jpg'
    s3.upload_file(output_filename, result_bucket_name, image_result_key)