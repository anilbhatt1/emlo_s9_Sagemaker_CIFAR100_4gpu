import sagemaker
import uuid

sagemaker_session = sagemaker.Session()
print('sagemaker version:' + sagemaker.__version__)
--
role = sagemaker.get_execution_role()
role
---
bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/emlo-s9-pt-cifar100'

checkpoint_suffix = str(uuid.uuid4())[:8]
checkpoint_s3_path = 's3://{}/checkpoint-{}'.format(bucket, checkpoint_suffix)
#checkpoint_s3_path = 's3://sagemaker-ap-south-1-426011120934/checkpoint-b56bdf03'

print(f'checkpointing path : {checkpoint_s3_path}')
--
import os
import subprocess 

instance_type = 'local'

if subprocess.call('nvidia-smi') == 0:
    # Set type to GPU if one is present
    instance_type = 'local_gpu'
    
print('instance_type:', instance_type)
---
pip install pytorch-lightning --quiet
---
import numpy as np
import torchvision, torch
import matplotlib.pyplot as plt
import pickle
from cifar100_pl_v0 import CIFAR100DataModule

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
--
# get some random test images to see if testloader works fine & images are getting displayed correctly
dataiter = iter(testloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
labels_list = labels.tolist()
for i in range(len(labels_list)):
    print(labels_list[i], label_names[labels_list[i]] )
--

# get some random train images to see if trainloader works fine & images are getting displayed correctly
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
labels_list = labels.tolist()
print('Showing label names of first 8 train images')
for i in range(8):
    print(labels_list[i], label_names[labels_list[i]] )

---
inputs = sagemaker_session.upload_data(path = "data", bucket = bucket, key_prefix="data/cifar100")
--
from sagemaker.pytorch import PyTorch
--
use_spot_instances = True
max_run = 1800
max_wait = 1200 if use_spot_instances else None
--
from sagemaker.pytorch import PyTorch

hyperparameters = {'batch_size': 32, 'checkpoint-path':checkpoint_s3_path}

checkpoint_local_path="/opt/ml/checkpoints"

cifar100_estimator = PyTorch(
    entry_point = "cifar100_pl_v0.py",
    role=role,
    framework_version="1.7.1",
    py_version="py3",
    hyperparameters=hyperparameters,
    instance_count=1,
    instance_type="ml.p3.8xlarge",
    base_job_name = 'cifar100-pytorch-Jan20-v0-2022-spot',
    checkpoints_s3_uri = checkpoint_s3_path,
    checkpoint_local_path = checkpoint_local_path,
    debugger_hook_config = False,
    use_spot_instances = use_spot_instances,
    max_run = max_run,
    max_wait = max_wait
)
--
cifar100_estimator.fit(inputs)
--
from sagemaker.pytorch import PyTorchModel

predictor = cifar100_estimator.deploy(initial_instance_count=1, instance_type="ml.c4.8xlarge")
---
# get some test images

dataiter = iter(testloader)
images, labels = dataiter.next()
print(images.size())

# print images
imshow(torchvision.utils.make_grid(images))
print('Ground Truth :', ' '.join('%4s' % classes[labels[j]] for j in range(4)))

outputs = predictor.predict(images.numpy())

_, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)

print('Predicted : ', ' '.join('%4s' % classes[predicted[j]] for j in range(4)))
--

predictor.delete_endpoint()  #Very Important !!!

--