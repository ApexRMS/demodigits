import rasterio
from sklearn.datasets import load_digits

digits = load_digits(as_frame=True)

for img in range(0, len(digits.images)):
    with rasterio.open(f"../data/digit_{img:04}.tif", mode="w",
                       driver="GTiff", width=digits.images[0].shape[0],
                       height=digits.images[0].shape[1], count=1, 
                       dtype=digits.images[0][0][0].dtype) as infile:
        infile.write(digits.images[img], indexes=1)

# Add targets to the data folder
digits.target.to_csv("../target.csv")
        
# Tar zip in the command prompt using tar cvzf data.tar.gz *.tif in data folder
# Tar zip targets using tar cvzf targets.tar.gz target.csv in data folder
        