import pandas as pd
import numpy as np
import random
import statistics
import math
import boto3
import warnings
warnings.filterwarnings("ignore")
from io import BytesIO
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from sklearn.preprocessing import LabelEncoder

def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            if "=" in line:
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip()
    return config


def list_datasets(bucket, prefix):
    # Ensure the prefix ends with a '/'
    if not prefix.endswith('/'):
        prefix += '/'

    try:
        # List objects within the bucket for a specific prefix
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        file_list = []
        if 'Contents' in response:
            for item in response['Contents']:
                file_name = item['Key'][len(prefix):]
                # Append the file name to the list, only if it is not empty or a folder placeholder
                if file_name and not file_name.endswith('/'):
                    file_list.append(file_name)
        
        return file_list

    except ClientError as e:
        print(f"An error occurred: {e}")
        return []
    
def download_object_as_dataframe(bucket_name, object_key):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        # Read the object (CSV file) as a DataFrame
        data_frame = pd.read_csv(BytesIO(response['Body'].read()))

        # print(f"Successfully downloaded object '{object_key}' from bucket '{bucket_name}' as a DataFrame.")
        return data_frame
    except ClientError as e:
        if e.response['Error']['Code'] == "403":
            print(f"Do not have permissions to download object '{object_key}' from bucket '{bucket_name}'.")
        elif e.response['Error']['Code'] == "NoSuchKey":
            print(f"The object '{object_key}' does not exist in bucket '{bucket_name}'.")
        else:
            print(f"Error downloading object '{object_key}' from bucket '{bucket_name}':")
            print(e)
        return None

# Assume the join key is a single column
def load_dm_from_s3(dm, dirs, bucket_name, join_key, join_key_domain):
    for data in dirs:
        df = download_object_as_dataframe(bucket_name, 'raw_data/' + data)
        if join_key not in df.columns:
            # Step 1: Identify the jk column
            jk_columns = [col for col in df.columns if join_key in col]
            # Assuming there is only one such column
            if jk_columns:
                for jk_column in jk_columns:
                    pattern = r'^\d{2}[A-Za-z]\d{3}$'  # Adjust the pattern if the format assumption is different
                    if df[jk_column].astype(str).str.match(pattern).all():
                        # Step 3: Rename the Column
                        df.rename(columns={jk_column: join_key}, inplace=True)
                        break
        aggdata = agg_dataset()
        aggdata.load(df)
        dm.add_seller(
            aggdata.data, data, [[join_key]], 
            join_key_domain, aggdata.X
        )
    
def remove_outliers(df, column):
    # Normalize the column
    mean = df[column].mean()
    std = df[column].std()
    normalized = (df[column] - mean) / std
    
    Q1 = normalized.quantile(0.25)
    Q3 = normalized.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create a boolean mask for rows to keep
    mask = (normalized >= lower_bound) & (normalized <= upper_bound)
    
    # Apply the mask to the dataframe
    df_filtered = df[mask]
    
    return df_filtered
    
class Preprocess:
    # load data (in the format of dataframe)
    # user provides dimensions (these dimensions will be pre-aggregated)
    # name should be unique across datasets
    def load(self, data, join_key, divide_by_max=True):
        self.data = data
        self.join_key = join_key
        self.X = []
        self.find_features()
        self.to_numeric_and_impute_all(divide_by_max)
    
    def agg_by_jk(self):
        self.data = self.data.groupby(
            self.join_key)[self.X].mean().reset_index()
    
    # iterate all attributes and find candidate features
    def find_features(self):
        atts = []
        for att in self.data.columns:
            if att != self.join_key and self.is_feature(att, 0.6, 20):
                atts.append(att)
                
        self.X = atts
        
    def is_feature(self, att, pct, unique_val):
        self.to_numeric(att)
        col = self.data[att]
        missing = sum(np.isnan(col))/len(self.data)
        distinct = len(col.unique())
        mean_value = col.mean()
        
        if missing < pct and distinct > unique_val and not np.isinf(
            mean_value):
            return True
        else:
            return False
        
    # this is the function to transform an attribute to number
    def to_numeric(self, att):
        # parse attribute to numeric
        self.data[att] = pd.to_numeric(self.data[att],errors="coerce")
    
    def impute_mean(self, att):
        mean_value=self.data[att].mean()
        self.data[att].fillna(value=mean_value, inplace=True)
        
    def to_numeric_and_impute_all(self, divide_by_max):
        new_X = []
        for att in self.X:
            self.to_numeric(att)
            self.impute_mean(att)
            if divide_by_max:
                self.data[att] /= np.abs(self.data[att].values).max()
                # self.data[att] /= self.data[att].mean()
            if self.data[att].std() > 0.1:
                new_X.append(att)
        self.X = new_X
        

config = read_config('keys.private')
access_key = config.get("public_key")
secret_access_key = config.get("secret_key")
# Initialize a session using the credentials and region
session = boto3.Session(region_name='us-east-2')
s3 = session.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_access_key,
)
