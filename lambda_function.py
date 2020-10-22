import json
import urllib.parse
import random
import boto3
import tempfile
import datetime
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

prefix = "res-"
suffix = ".jpeg"

s3_client = boto3.client('s3')
s3 = boto3.resource('s3')
resized_bucket = 'wc2ai-resized'
model_bucket = 'wc2lambdacode'

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('test_ds')
saved_pref = 's3://PutPhotoBuccket/'
model_path = 's3://codeBucket/converted_model.tflite'
model_key = 'converted_model.tflite'
resized_path = '/tmp/resized.jpeg'

width = 456
height = 456


def classify_image(image_arr):
    tmp4 = tempfile.NamedTemporaryFile()
    with open(tmp4.name, 'wb') as modelFile:
        key4 = urllib.parse.unquote_plus(model_key)
        s3.Bucket(model_bucket).download_file(key4, tmp4.name)
        tmp4.flush()

        try:
            interpreter = tflite.Interpreter(model_path=tmp4.name)
            interpreter.allocate_tensors()

            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            input_data = np.array(image_arr, dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], [input_data])
            interpreter.allocate_tensors()
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred_class = output_data.argmax()
            return pred_class

            print("classify well done!")
        except Exception as e:
            print(e)
            print('Error while classifying image')

        # print(result)
        # return result


def saveto_dynamo(table, filename, calculated_class):
    currentDT = datetime.datetime.now()
    formattedDT = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    formattedClass = int(calculated_class)
    requestId = str(random.randint(1, 10000))
    filePath = saved_pref + "" + filename
    table.put_item(
        Item={
            'reqid': requestId,
            'fpath': filePath,
            'dt': formattedDT,
            'class': formattedClass,
        }
    )
    print("Saved in DynamoDB: %s , %s , %s , %d" % (formattedDT, requestId, filePath, formattedClass))


def lambda_handler(event, context):
    # print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    try:
        print('Running Deep Learning example using Tensorflow library ...')
        print('Image to be processed, from: bucket [%s], object key: [%s]' % (bucket, key))

        # load image
        tmp = tempfile.NamedTemporaryFile()

        with open(tmp.name, 'wb') as trainFile:
            s3.Bucket(bucket).download_file(key, tmp.name)
            tmp.flush()
            print(tmp.name)

            img = Image.open(tmp.name).resize((width, height))
            print(img.format, img.size, img.mode)
            img.save(resized_path, "JPEG")

            img_arr = np.array(img)
            img_arr = img_arr.astype(np.float) / 255

            # Classify image
            try:
                calc_class = classify_image(img_arr)
            except Exception as e:
                print(e)
                print('Error while image prepare for classify')

            # Upload the file
            file_in_bucket = prefix + "" + key
            try:
                response = s3_client.upload_file(resized_path, resized_bucket, file_in_bucket)
            except Exception as e:
                print(e)
                print('Error while uploading resized file')

                # save into DynamoDB
            try:
                saveto_dynamo(table, file_in_bucket, calc_class)
            except Exception as e:
                print(e)
                print('Error while save to DynamoDB')

            print('Finished!')


    except Exception as e:
        print(e)
        print(
            'Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(
                key, bucket))
        raise e
