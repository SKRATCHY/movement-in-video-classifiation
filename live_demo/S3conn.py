import boto3
import pandas as pd
import pickle
import os

s3 = boto3.resource(service_name = "s3")
bucket = s3.Bucket(os.getenv('s3_test_bucket'))
s3_client = boto3.client("s3")

def find_files(file_name, s3_bucket=bucket):
    files = []
    found = False
    for my_bucket_object in s3_bucket.objects.all():
        if str(my_bucket_object.key).__contains__(file_name):
            found = True
            files.append(my_bucket_object.key)
    if not found:
        print("No files found with that keyword")
    return(files)

def find_file(file_name, s3_bucket=bucket):
    files = []
    found = False
    for my_bucket_object in s3_bucket.objects.all():
        if str(my_bucket_object.key) == file_name:
            found = True
            files.append(my_bucket_object.key)
    if not found:
        print("No files found with that keyword")
    return(files)

def get_video_url(video_key, s3_bucket=bucket, client=s3_client):
    found = False
    video_url = []
    for my_bucket_object in s3_bucket.objects.all():
        if str(my_bucket_object.key) == video_key:
            found = True
            video_url = client.generate_presigned_url('get_object',Params = {'Bucket' : s3_bucket.name ,'Key' : str(my_bucket_object.key)}, ExpiresIn=600)
    if not found:
        print("Video not found")    
    return(video_url)

def delete_file(file_name, s3_bucket=bucket, client=s3_client):
    found = False
    for my_bucket_object in s3_bucket.objects.all():
        if my_bucket_object.key == file_name:
            found = True
            client.delete_object(Bucket=s3_bucket.name, Key=file_name)
    if not found:
        print("No files found with that keyword")

def upload_file(file, output_name, s3_bucket=bucket):
    s3_client.upload_file(file, s3_bucket.name , output_name)
    print("file uploaded")

def get_model(modelo):
    model = pickle.loads(bucket.Object(modelo).get()['Body'].read())
    return(model)


#Retrieves a csv file stored in S3
def get_csv(csv_path, s3_bucket=bucket):
    if csv_path.split(".")[-1] != "csv":
        print("Please send a csv file as parameter")
        return([])
        
    for my_bucket_object in s3_bucket.objects.all():
        if str(my_bucket_object.key) == csv_path:            
            response = s3_client.get_object(Bucket=s3_bucket.name, Key=csv_path)
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 200:
                df = pd.read_csv(response.get("Body"))
                return(df)
    print("File not found")
    return([])