#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as msc
import imageio as imio
import glob
import matplotlib.image as matimg
from skimage import io, color
from sklearn.metrics import classification_report
import csv
from PIL import Image

#
from sklearn.linear_model import LogisticRegression

#We import the library for the model


# In[2]:


images = []
file = "data/data/*"
for photo in glob.glob(file):
    image = io.imread(photo)
    image = color.rgb2gray(image)
# We ,,extract" the color from the image so we can
# have a 2-D vector instead of a 3-D 
# The image shape becomes from (224,224,3) => (224,224)
    images.append(image)
print(np.shape(images))
images = np.reshape(images,(-1,224*224))
#We reshape the array of the images to make it 2-D instead of 3-D 
#=> from (no_of_images,224,224)
# to => (no_of_images,224*224)
print(np.shape(images))


# In[3]:


print("Da")


# In[4]:


train_labels = np.loadtxt('train_labels.txt',delimiter = ',',skiprows=1)
#We load the data from the txt documents, we skip the first row which is "id, class"
#so we can get directly to the photo_name and what class it belongs to

# In[5]:


train_labels = np.ravel(train_labels)
train_labels = train_labels[1::2]
# For training the model we only need the classes, not the photo name
# So we know that the classes are on odd position, so
# now we have the array [class1,class2,class3,...]
print(np.shape(train_labels))


# In[6]:


model = LogisticRegression(solver='lbfgs', max_iter=15000)
# We initialize the model

# In[7]:


print(np.shape(train_labels))
print(np.shape(images))


# In[8]:


model.fit(images[0:15000],train_labels[0:15000])
#We train the model with the first 15.000 images

# In[9]:


print(model.score(images[15001:17000],train_labels[15001:17000]))
# We get the score, so we know aproximately where are we situated with the accuracy

# In[10]:


aray = model.predict(images[17000:22151])
print(len(images[17000:22151]))
# After the model is trained,
# We predict what classes the images that are left belong to
# So in aray[0] we will have the class that photo 17001 belongs to
# In array[1] we will have the class that photo 17002 belongs to
...

# In[11]:


print(len(aray))


# In[12]:


#We print the result in a csv file
# We print the name of the photo and the class that is predicted to it from the aray


with open('rezultat.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id","class"])
    for i in range(len(aray)):
        x = int(17001) + int(i)
        writer.writerow(["0"+str(x),int(aray[i])])


# In[13]:


#with open('rezultat.csv', 'r') as f: 
 #   lines = f.readlines() 
#with open('rezultat.csv', 'w') as f: 
 #   f.writelines(lines[:-1]) 


# In[ ]:





# In[ ]:




