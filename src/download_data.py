# Download data from GitHub, extract TIF files from data.tar.gz

import pysyncrosim as ps
import pandas as pd
import requests
import tarfile
import os
import io

# Delete later:
# import tempfile

# Get environment
env = ps.environment._environment()
transfer_dir = env.transfer_directory.item()
temp_dir = ps.environment.runtime_temp_folder("tifs")

# Get the Scenario that is currently being run
my_scenario = ps.Scenario()

# Download TIF files and extract
url = "https://raw.github.com/ApexRMS/demodigits/main/data.tar.gz"
data = os.path.join(transfer_dir, "data.tar.gz")
request = requests.get(url, stream=True)

with open(data, "wb") as f:
    f.write(request.raw.read())
    
# Download and extract target csv
url = "https://raw.github.com/ApexRMS/demodigits/main/target.csv"
request = requests.get(url)
target = pd.read_csv(io.StringIO(request.content.decode('utf-8')))

# Initialize X, y 
X = []
y = []

# Write data and targets to X, y
with tarfile.open(data, "r:gz") as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar, path=temp_dir)
    extracted_imgs = os.listdir(temp_dir)
    for count, img in enumerate(extracted_imgs):
        X.append(os.path.join(temp_dir, img))
        y.append(target.target.iloc[count])
        
# Save X and y to an intermediate datasheet
input_df = pd.DataFrame({"X": X, "y": y})
my_scenario.save_datasheet(name="InputData", data=input_df)
