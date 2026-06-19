#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.optimizers import Adam


# In[60]:


import cv2

video_path = "dog_video.mp4"

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

if not ret:
    print("Failed to read frame")
else:
    print(frame.shape)


# In[61]:


from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

model = MobileNet(weights="imagenet")

img = cv2.resize(frame, (224,224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

results = model.predict(img)

print(decode_predictions(results, top=5)[0])


# In[62]:


from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model(frame)

results[0].show()


# while True:
#     ret, frame = cap.read()
# 
#     if not ret:
#         break
# 
#     results = model(frame)
# 
# cap.release()

# In[64]:


imu = pd.read_csv("collar_imu.csv")


# In[65]:


import numpy as np
import pandas as pd

imu = pd.read_csv("collar_imu.csv")

imu["mag"] = np.sqrt(
    imu["accel_x"]**2 +
    imu["accel_y"]**2 +
    imu["accel_z"]**2
)


# In[66]:


print(imu.head())


# In[67]:


print(imu.columns)


# In[68]:


print(imu.columns)


# In[69]:


imu["window"] = imu["timestamp_ms"] // 500

imu_summary = (
    imu.groupby("window")
       .agg(
           timestamp_ms=("timestamp_ms", "first"),
           mean_mag=("mag", "mean")
       )
       .reset_index(drop=True)
)

print(imu_summary.head())


# In[70]:


threshold = 1.8

imu_summary["imu_activity"] = np.where(
    imu_summary["mean_mag"] > threshold,
    "Active",
    "Static"
)

print(imu_summary)


# In[71]:


vision_activity = "Static"
vision_conf = 0.55


# In[72]:


imu_mag = imu_summary.loc[0, "mean_mag"]

threshold = 1.8

if imu_mag > threshold:
    final = "Active"
else:
    final = vision_activity

print(final)


# In[73]:


max_mag = imu_summary["mean_mag"].max()

imu_conf = imu_mag / max_mag

confidence = max(vision_conf, imu_conf)

print(confidence)


# In[74]:


#json


# In[75]:


import json

output = [
    {
        "timestamp_ms": 0,
        "activity": "Static",
        "confidence": 0.82
    },
    {
        "timestamp_ms": 500,
        "activity": "Active",
        "confidence": 0.91
    }
]

with open("timeline.json", "w") as f:
    json.dump(output, f, indent=4)

print("timeline.json created successfully!")


# In[76]:


[
  {
    "timestamp_ms": 0,
    "activity": "Static",
    "confidence": 0.82
  },
  {
    "timestamp_ms": 500,
    "activity": "Active",
    "confidence": 0.91
  },
  {
    "timestamp_ms": 1000,
    "activity": "Active",
    "confidence": 0.95
  }
]


# In[77]:


output = []


# In[78]:


timestamp = 0
activity = "Static"
confidence = 0.82


# In[79]:


output.append({
    "timestamp_ms": timestamp,
    "activity": activity,
    "confidence": round(confidence, 2)
})


# In[ ]:





# In[ ]:





# In[80]:


number_of_samples = 12 * 2
print(number_of_samples)   # 24


# In[81]:


output = []

number_of_samples = 24

for i in range(number_of_samples):
    timestamp = i * 500          # 0, 500, 1000, ...
    activity = "Static"          # Replace with your predicted activity
    confidence = 0.80            # Replace with your computed confidence

    output.append({
        "timestamp_ms": timestamp,
        "activity": activity,
        "confidence": round(confidence, 2)
    })

import json

with open("timeline.json", "w") as f:
    json.dump(output, f, indent=4)

print("timeline.json created successfully!")


# In[82]:


[
    {
        "timestamp_ms": 0,
        "activity": "Static",
        "confidence": 0.8
    },
    {
        "timestamp_ms": 500,
        "activity": "Static",
        "confidence": 0.8
    },
    {
        "timestamp_ms": 1000,
        "activity": "Static",
        "confidence": 0.8
    }
]

