# Databricks notebook source
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
%matplotlib inline

# COMMAND ----------

with open("/dbfs/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/intersect_list.txt","r") as f:
    intersect_list = f.read()
    intersect_list = intersect_list.splitlines()

# COMMAND ----------

rand_idx = np.random.randint(0,len(intersect_list))
#rand_idx =

base_path = '/dbfs/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/FLIR_ADAS_1_3/train/'
rgb_path = base_path+'RGB/'+intersect_list[rand_idx]+'.jpg'
therm16_path = base_path+'thermal_16_bit/'+intersect_list[rand_idx]+'.tiff'
therm8_path = base_path+'thermal_8_bit/'+intersect_list[rand_idx]+'.jpeg'

file_paths = [rgb_path,therm16_path,therm8_path]
fig = plt.figure(figsize=(20, 20))
columns = 3
rows = 1
for i in range(1, columns*rows+1):
    img = Image.open(file_paths[i-1])
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

# COMMAND ----------


