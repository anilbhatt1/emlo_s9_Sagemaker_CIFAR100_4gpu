# Training CIFAR100 on AWS Sagemaker with 4 GPUs using Pytorch Lightning

### Steps to be followed

- Train CIFAR100 in google colab and save the weights for future use while using AWS sagemaker.
  - Notebook Reference : https://github.com/anilbhatt1/emlo_s9_Sagemaker_CIFAR100_4gpu/blob/master/emlo_s9_cifar100_tensorboard_resnet34_v0.ipynb
  - GPU weights can be found in https://drive.google.com/file/d/1bzAdT-oLkr8EDMWflCaIIKySR-M8CI6f/view?usp=sharing
  - CPU weights can be found in https://drive.google.com/file/d/1HYqZo1P-1v0Pdmccc4L1-hIrPgI79Xuj/view?usp=sharing
- Create AWS notebook instance. Used ml.t2.medium for creating notebook instance.
- Create a new IAM role with access to any s3 bucket and attach it to AWS notebook instance. This can be done while creating notebook instance itself.
- Create a jupyter notebook inside the notebook instance. Used conda_pytorch_latest_p36 while creating notebook.
- Do necessary imports
- Prepare a python script. Training is done by executing this script through a sagemaker estimator created via Jupyter notebook. 
  - Script Reference : https://github.com/anilbhatt1/emlo_s9_Sagemaker_CIFAR100_4gpu/blob/master/cifar100_pl_v2.py
  - Template jupyter notebook reference (that was created using conda_pytorch_latest_p36) : https://github.com/anilbhatt1/emlo_s9_Sagemaker_CIFAR100_4gpu/blob/master/sagemaker_notebook%20_CIFAR100.py
  - Reference to use checkpoint : https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html
- Create an estimator with PyTorch class from sagemaker.pytorch library.
- Whole pytorch lightning code will be inside the py script referred above. Trainer is defined inside py script and also fitted inside the same.
- Train with required instance by executing fit method available for sagemaker estimator. ml.p3.8xlarge was used for training. Spot training was used as it will give around 70% cost saving.
- Create a predictor with same estimator. **deploy** method available for sagemaker estimator was used for the same.  This will create an endpoint that we can find from the **inference** section in AWS sagemaker page.
- Predictor was created with ml.c4.8xlarge instance. This is a CPU only instance.
- Using **predict** method available for sagemaker estimator, inferencing was done for the CIFAR100 images supplied via testloader. **predict** method calls **model_fn** available in the py script. **model_fn** loads the CPU model saved in **opt/ml/model** path (saved using **save_model** function in script) and uses it for inferencing.
- Execute **predictor.delete_endpoint()** once purpose is served so that endpoint created can be deleted.

### Youtube video link of training 

- https://youtu.be/PxnEH681aAw
