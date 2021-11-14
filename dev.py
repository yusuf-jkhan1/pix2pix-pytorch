# Databricks notebook source
!pip install --upgrade Pillow --global-option="build_ext" --global-option="--enable-tiff" --global-option="--enable-jpeg"

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
%matplotlib inline

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls '/dbfs/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/'
# MAGIC echo -----------------
# MAGIC ls '/dbfs/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/FLIR_ADAS_1_3/'
# MAGIC echo -----------------
# MAGIC ls '/dbfs/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/FLIR_ADAS_1_3/train/'
# MAGIC echo -----------------
# MAGIC ls '/dbfs/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/FLIR_ADAS_1_3/train/RGB' | head
# MAGIC echo -----------------
# MAGIC ls '/dbfs/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/FLIR_ADAS_1_3/train/thermal_8_bit' | head

# COMMAND ----------

ls_rgb = dbutils.fs.ls("dbfs:/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/FLIR_ADAS_1_3/train/RGB")
ls_thermal_8 = dbutils.fs.ls("dbfs:/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/FLIR_ADAS_1_3/train/thermal_8_bit")
ls_thermal_16 = dbutils.fs.ls("dbfs:/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/FLIR_ADAS_1_3/train/thermal_16_bit")

# COMMAND ----------

rgb_list = [file.name for file in ls_rgb]
thermal_8_list = [file.name for file in ls_thermal_8]
thermal_16_list = [file.name for file in ls_thermal_16]

# COMMAND ----------

rgb_set = set([file.split(".")[0] for file in rgb_list])
thermal_8_set = set([file.split(".")[0] for file in thermal_8_list])
thermal_16_set = set([file.split(".")[0] for file in thermal_16_list])

# COMMAND ----------

intersect = rgb_set.intersection(thermal_8_set).intersection(thermal_16_set)

# COMMAND ----------

with open("/dbfs/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/intersect_list.txt","w") as f:
    for i in list(intersect):
        f.write(i)
        f.write("\n")

# COMMAND ----------

ntxn

# COMMAND ----------

file_path1 = '/dbfs/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/FLIR_ADAS_1_3/train/RGB/FLIR_01574.jpg'
file_path2 = '/dbfs/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/FLIR_ADAS_1_3/train/thermal_16_bit/FLIR_01574.tiff'
file_path3 = '/dbfs/mnt/r&d_trial/R&D/IntelligentSystems/PublicData/FlirADAS/FLIR_ADAS_1_3/train/thermal_8_bit/FLIR_01574.jpeg'

# COMMAND ----------

pil_im1 = Image.open(file_path1)
pil_im2 = Image.open(file_path2)
pil_im3 = Image.open(file_path3)

# COMMAND ----------

plt.imshow(pil_im1)

# COMMAND ----------

plt.imshow(pil_im2, cmap='gray')

# COMMAND ----------

plt.imshow(pil_im3, cmap='gray')

# COMMAND ----------

pil_im3 = np.array(pil_im3)
pil_im3 = pil_im3**1.4
plt.imshow(pil_im3, cmap='gray')

# COMMAND ----------


