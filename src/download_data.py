# Download data from GitHub, extract TIF files from data.tar.gz

import pysyncrosim as ps
import rasterio
import pandas as pd
import requests
import tarfile
import os
import io

# Delete later:
import tempfile

# Get environment
env = ps.environment._environment()
transfer_dir = env.transfer_directory

# Get the Scenario that is currently being run
my_scenario = ps.Scenario()

# Get the input Datasheet?

# Download TIF files and extract
url = "https://raw.github.com/ApexRMS/demodigits/main/data.tar.gz"
data = "data.tar.gz"
request = requests.get(url, stream=True)

with open(data, "wb") as f:
    f.write(request.raw.read())
    
# Download and extract target csv
url = "https://raw.github.com/ApexRMS/demodigits/main/target.csv"
request = requests.get(url)
target = pd.read_csv(io.StringIO(request.content.decode('utf-8')))
target.target.iloc[0]

# for testing, delete later:
transfer_dir = tempfile.mkdtemp()

# Initialize X, y 
X = []
y = []

# Write data and targets to X, y
with tarfile.open(data, "r:gz") as tar:
    tar.extractall(path=transfer_dir)
    extracted_imgs = os.listdir(transfer_dir)
    for count, img in enumerate(extracted_imgs):
        with rasterio.open(os.path.join(transfer_dir, img), "r") as raster:
            values = raster.read()
            if values.shape[0] == 1:
                X.append(values[0])
            else:
                X.append(values)
        y.append(target.target.iloc[0])
        
# Save X and y to an intermediate datasheet
                
            
    

