#!/usr/bin/env python
# coding: utf-8

# ## Pencil Sketch with Python

# In[1]:


# Import Libraries 
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


# Load your Images
image = cv2.imread("C:\\Users\\aishw\\Downloads\\images.jpeg")
plt.imshow(image)


# In[3]:


img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(img)


# In[4]:


# Convert Your Picture To Grayscale
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')


# In[5]:


# Smooth your picture
blur = cv2.GaussianBlur(gray_image,(5,5),2)
plt.imshow(blur,cmap='gray')


# In[6]:


smoothed = cv2.GaussianBlur(gray_image,(7,7),5)
plt.imshow(smoothed)


# In[7]:


#Edge detection
sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5) # Change in horizonal direction, dx
sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=5) # Change in verticle direction, dy
gradmag_sq = np.square(sobelx)+np.square(sobely) # Square the images element-wise and then add them together 
gradmag = np.sqrt(gradmag_sq) # Take the square root of the resulting image element-wise to get the gradient magnitude

plt.imshow(gradmag, cmap ='gray')


# In[8]:


gradmag_inv = 255-gradmag
plt.imshow(gradmag_inv, cmap='gray')


# In[9]:


thresh_value, thresh_img = cv2.threshold(gradmag_inv,10,255,cv2.THRESH_BINARY)
plt.imshow(thresh_img, cmap = 'gray')


# In[10]:


pencilsketch_gray, pencilsketch_color  = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05) 

plt.imshow(pencilsketch_gray, cmap ='gray')


# In[ ]:





# In[ ]:




