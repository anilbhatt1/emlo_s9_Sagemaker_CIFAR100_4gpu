#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sagemaker
import uuid

sagemaker_session = sagemaker.Session()
print('sagemaker version:' + sagemaker.__version__)


# In[2]:


role = sagemaker.get_execution_role()
role


# In[3]:


bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/emlo-s9-pt-cifar100_jan25'

checkpoint_suffix = str(uuid.uuid4())[:8]
checkpoint_s3_path = 's3://{}/checkpoint-{}'.format(bucket, checkpoint_suffix)


# In[4]:


print(f'checkpointing path : {checkpoint_s3_path}')


# In[5]:


import os
import subprocess 

instance_type = 'local'

if subprocess.call('nvidia-smi') == 0:
    # Set type to GPU if one is present
    instance_type = 'local_gpu'
    
print('instance_type:', instance_type)


# In[6]:


pip install pytorch-lightning --quiet


# In[7]:


pip install googledrivedownloader


# In[9]:


import numpy as np
import torchvision, torch
import matplotlib.pyplot as plt
import pickle
from cifar100_pl_v1 import CIFAR100DataModule

dm = CIFAR100DataModule(batch_size=32)
dm.prepare_data()
dm.setup()
trainloader = dm.train_dataloader()
testloader = dm.test_dataloader()

# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
#function to read files present in the Python version of the dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict

metaData = unpickle('./data/cifar-100-python/meta')
label_names = metaData['fine_label_names']
print(len(label_names))


# In[10]:


# get some random test images to see if testloader works fine & images are getting displayed correctly
dataiter = iter(testloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
labels_list = labels.tolist()
for i in range(len(labels_list)):
    print(labels_list[i], label_names[labels_list[i]] )


# In[11]:


# get some random train images to see if trainloader works fine & images are getting displayed correctly
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
labels_list = labels.tolist()
print('Showing label names of first 8 train images')
for i in range(8):
    print(labels_list[i], label_names[labels_list[i]] )


# In[12]:


# Checking if gdrive download works
from google_drive_downloader import GoogleDriveDownloader as gdd
sample_gdrive_path = './Resnet34_pl_cifar100.pt'
if not os.path.exists(sample_gdrive_path):
      gdd.download_file_from_google_drive(file_id='1bzAdT-oLkr8EDMWflCaIIKySR-M8CI6f',
                                          dest_path='./Resnet34_pl_cifar100.pt', unzip=False)


# In[13]:


inputs = sagemaker_session.upload_data(path = "data", bucket = bucket, key_prefix="data/cifar100")


# In[14]:


from sagemaker.pytorch import PyTorch


# In[15]:


use_spot_instances = True
max_run = 1200
max_wait = 1210 if use_spot_instances else None


# instance_count=2, instance_type="ml.p3.8xlarge", gpus=4, strategy = "ddp", num_nodes=2
# 
# Script : cifar100_pl_v1.py

# In[18]:


from sagemaker.pytorch import PyTorch

# The local path where the model will save its checkpoints in the training container
checkpoint_local_path="/opt/ml/checkpoints"

hyperparameters = {'batch_size': 32} 

cifar100_estimator = PyTorch(
    entry_point = "cifar100_pl_v1.py",
    role=role,
    framework_version="1.7.1",
    py_version="py3",
    hyperparameters=hyperparameters,
    instance_count=2,
    instance_type="ml.p3.8xlarge",
    base_job_name = 'cifar100-pytorch-Jan25-v1-2022-spot',
    checkpoint_s3_uri = checkpoint_s3_path,
    checkpoint_local_path=checkpoint_local_path,
    debugger_hook_config = False,
    use_spot_instances = use_spot_instances,
    max_run = max_run,
    max_wait = max_wait
)


# In[19]:


cifar100_estimator.fit(inputs)


# instance_count=1, instance_type="ml.p3.8xlarge", gpus=4, strategy = "ddp"
# 
# Script : cifar100_pl_v2.py

# In[20]:


from sagemaker.pytorch import PyTorch

# The local path where the model will save its checkpoints in the training container
checkpoint_local_path="/opt/ml/checkpoints"

hyperparameters = {'batch_size': 32} 

cifar100_estimator = PyTorch(
    entry_point = "cifar100_pl_v2.py",
    role=role,
    framework_version="1.7.1",
    py_version="py3",
    hyperparameters=hyperparameters,
    instance_count=1,
    instance_type="ml.p3.8xlarge",
    base_job_name = 'cifar100-pytorch-Jan25-v1-2022-spot',
    checkpoint_s3_uri = checkpoint_s3_path,
    checkpoint_local_path=checkpoint_local_path,
    debugger_hook_config = False,
    use_spot_instances = use_spot_instances,
    max_run = max_run,
    max_wait = max_wait
)


# In[21]:


cifar100_estimator.fit(inputs)


# In[22]:


from sagemaker.pytorch import PyTorchModel

predictor = cifar100_estimator.deploy(initial_instance_count=1, instance_type="ml.c4.8xlarge")


# In[24]:


dataiter = iter(testloader)


# In[26]:


# get some test images

images, labels = dataiter.next()
print(images.size())

# print images
imshow(torchvision.utils.make_grid(images))
print('Ground Truth :', ' '.join('%4s' % label_names[labels[j]] for j in range(4)))

outputs = predictor.predict(images.numpy())

_, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)

print('Predicted : ', ' '.join('%4s' % label_names[predicted[j]] for j in range(4)))


# Checking if checkpoint is picked up while running with same config during same instance
# 
# instance_count=1, instance_type="ml.p3.8xlarge", gpus=4, strategy = "ddp"
# 
# Script : cifar100_pl_v2.py

# In[27]:


cifar100_estimator.fit(inputs)


# In[28]:


# get some test images

images, labels = dataiter.next()
print(images.size())

# print images
imshow(torchvision.utils.make_grid(images))
print('Ground Truth :', ' '.join('%4s' % label_names[labels[j]] for j in range(4)))

outputs = predictor.predict(images.numpy())

_, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)

print('Predicted : ', ' '.join('%4s' % label_names[predicted[j]] for j in range(4)))


# In[29]:


predictor.delete_endpoint()  #Very Important !!!


# In[ ]:




