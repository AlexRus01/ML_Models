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
from sklearn.metrics import f1_score
from PIL import Image

#
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
#
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification

#We import the library for the model


# In[2]:


images = []
file = "data/data/*"
for photo in glob.glob(file):
    image = io.imread(photo)
    image = color.rgb2gray(image) #We ,,extract" the color from the image so we can
 					#have a 2-D vector instead of a 3-D => T
    images.append(image)	  # The image shape becomes from (224,224,3) => (224,224)
print(np.shape(images))
images = np.reshape(images,(-1,224*224))
print(np.shape(images))

#We reshape the array of the images to make it 2-D instead of 3-D => from (no_of_images,224,224)
			#					     to => (no_of_images,224*224)

# In[3]:


train_labels = np.loadtxt('train_labels.txt',delimiter = ',',skiprows=1)
#We load the data from the txt documents, we skip the first row which is "id, class"
#so we can get directly to the photo_name and what class it belongs to

validation_labels = np.loadtxt('validation_labels.txt',delimiter = ',',skiprows=1)

train_labels = np.ravel(train_labels) #We transform the matrix to a vector
				      # from [[photo_name1,class1], [photo_name2,class2]]
				      # to [photo_name1,class1,photo_name2,class2...]
validation_labels = np.ravel(validation_labels)
train_labels = train_labels[1::2]	
# For training the model we only need the classes, not the photo name
# So we know that the classes are on odd position, so
# now we have the array [class1,class2,class3,...]

validation_labels = validation_labels[1::2]
print(np.shape(train_labels))
print(np.shape(validation_labels))


# In[4]:


model = LogisticRegression(solver='lbfgs', max_iter=15000)
model3 = NearestCentroid()  # We initialize the model

print(np.shape(train_labels))
print(np.shape(images))

print("Da")


# In[5]:


#model.fit(images[0:15000],train_labels[0:15000])
model3.fit(images[0:15000],train_labels[0:15000])

#We train the model with the first 15.000 images

# In[6]:


#print(model.score(images[15001:17001],validation_labels))
model3.score(images[15001:17001],validation_labels)
# We get the score, so we know aproximately where are we situated with the accuracy


# In[7]:


#aray = model.predict(images[17000:22151])
aray = model3.predict(images[17000:22151])
# After the model is trained,
# We predict what classes the images that are left belong to
# So in aray[0] we will have the class that photo 17001 belongs to
# In array[1] we will have the class that photo 17002 belongs to
...


# In[8]:


testf1 = model3.predict(images[15000:17000])


print(f1_score(validation_labels,testf1))
#With the f1_score from the sklearn.metrics
# we can test the precision of our model
# using the validation_labels (from data) for the images 15000 to 17000

# In[9]:

#We print the result in a csv file
# We print the name of the photo and the class that is predicted to it from the aray

with open('rezultat.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id","class"])
    for i in range(len(aray)):
        x = int(17001) + int(i)
        writer.writerow(["0"+str(x),int(aray[i])])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




