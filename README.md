# Serving a Wide & Deep Learning model in Google Cloud Platform and Kubernetes

This guide consists of two main parts:
1. Creating and training a wide & deep learning model, and exporting it
2. Containerizing the model and making it available for serving in the cloud

Throughout this guide, I highly recommend you follow these 2 links:
- Vitaly Bezgachev's awesome posts, [Part 1](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-1-make-your-model-ready-for-serving-776a14ec3198), [Part 2](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-2-containerize-it-db0ad7ca35a7) and [Part 3](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-3-into-the-cloud-7115ff774bb6)
- [Tensorflow's Inception Serving guide](https://www.tensorflow.org/serving/serving_inception)

as I myself followed them very closely when deploying my own model.

## Tools you will need:

The main tools used to put the model to production are:

- TensorFlow for creating the wide and deep model and exporting it
- Docker for containerization
- TensorFlow Serving for model hosting
- Kubernetes in GCP for production deployment

The tools you will need may differ depending on the OS you are working on, and depending on whatever OS version you are in you may have to find workarounds for some of them. For me, as I am using Windows 10 Home Edition, there are some limitations such as no hyper-v virtualization and etc that I had to find workarounds. For me I used:

- Docker Toolbox
- Tensorflow 1.4.0
- Python 3.5
- Google Cloud SDK 
- A Google Account with billing enabled (check that you have the $300 free credits before trying out, this project is billable just a warning)
- Virtualization tool, for me I had Oracle Virtualbox

## 1. Creating and training a wide & deep learning model, and exporting it

Much of the concept of a wide & deep learning model is explained in the original paper [here](https://www.google.com.sg/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjtpqS7ntzXAhUIR48KHQ6yBkwQFggnMAA&url=https%3A%2F%2Farxiv.org%2Fabs%2F1606.07792&usg=AOvVaw0Ou4rDtQZAXCC1QqxJPBIu) as well in the guide [here](https://www.tensorflow.org/tutorials/wide_and_deep), so I won't go into the specifics of it. Some of the stuff you may want to read up on your own will be 
- creating the wide and deep columns
- creating the cross columns
- creating the columns with embedddings
- training and evaluating the model

This model used in this case is highly referenced from yufengg's [model](https://github.com/yufengg/widendeep), although I updated the APIs for the features according to the ones used in the wide & deep learning guide by Tensorflow. 

This example uses [Kaggle Criteo's Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge) for training the recommendation model and it outputs a prediction of 1 or 0 whether an ad will be clicked or not. The dataset consists of two files, train.csv and eval.csv. The dataset consists of a ground truth label, 26 categorical variables and 13 interval variables.

I have created a python file _train_and_eval.py_ that trains a wide and deep model using the _tf.contrib.learn.DNNLinearCombinedClassifier_ function. However, before you run the model, a key line in the code you will need to change is the line
```
export_folder = m.export_savedmodel("D:\Git Repo\wide-and-deep\models\model_WIDE_AND_DEEP_LEARNING" + '\export',serving_input_fn)
```
For me I used an an absolute path, so please do change the path in this line to your own absolute or relative path. 
This line creats a folder export inside the model folder, which contains these 2 things important for serving the tensorflow model:
1. a saved_model.pb
2. a variables folder

To run the model just go to the directory where the _train_and_eval.py_ as well as the _data_files_ folder is in and type
```
python train_and_eval.py
```
The model takes in the train.csv to train the model and the eval.csv to evaluate the model. It will take some time depending on how fast your computer processes. If your _data_files_ folder is located in another location, change these 2 lines:
```
train_file = "data_files/train.csv"
eval_file  = "data_files/eval.csv"
```
to point it towards the correct place instead.

It will then output a folder _models_ which will contain the model folder *model_WIDE_AND_DEEP_LEARNING*. The model folder contains all the checkpoints for the model. If you were to rerun the model again at a later time it will restore the previous checkpoints and train again from there. 

Do check that the saved_model.pb and variables folder is created from the run, and this will conclude the exporting of a trained model from Tensorflow. Next will be the containerization and serving it in the cloud.

## 2. Containerizing the model and making it available for serving in the cloud

For a Windows user, before you start on this part, remember to install Docker/Docker Toolbox(if you do not have Hyper-V).
For Docker Toolbox users, the docker commands are not readly available in the command line. Here are some simple steps to enable it.

Step 1. Run this command in your CLI
```
docker-machine env <name-of-your-docker-image> 
```
For mine it was 
```
docker-machine env default
```
which I checked from my Oracle VirtualBox.

Step 2. These lines will then appear:
```
SET DOCKER_TLS_VERIFY=1
SET DOCKER_HOST=tcp://192.168.99.100:2376
SET DOCKER_CERT_PATH=C:\Users\User\.docker\machine\machines\default
SET DOCKER_MACHINE_NAME=default
SET COMPOSE_CONVERT_WINDOWS_PATHS=true
REM Run this command to configure your shell:
REM     @FOR /f "tokens=*" %i IN ('docker-machine env default') DO @%i
```
Copy
```
@FOR /f "tokens=*" %i IN ('docker-machine env default') DO @%i
```
into the CLI and press enter. This will then enable docker commands in your shell.



## Credits and Useful Links (I'm spamming abit but that's how many links I referenced):

1. Vitaly Bezgachev's awesome posts, [Part 1](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-1-make-your-model-ready-for-serving-776a14ec3198), [Part 2](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-2-containerize-it-db0ad7ca35a7) and [Part 3](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-3-into-the-cloud-7115ff774bb6). He used Azure but I had some issues with Azure cli so I used GCP instead.
2. [Tensorflow's Inception Serving guide](https://www.tensorflow.org/serving/serving_inception)
3. Yufengg's [widendeep repo](https://github.com/yufengg/widendeep) which I got much of the wide and deep code from
4. Tensorflow's [Wide and Deep Tutorial](https://www.tensorflow.org/tutorials/wide_and_deep)
5. Weimin Wang's [blog post](https://weiminwang.blog/2017/09/12/introductory-guide-to-tensorflow-serving/) for Tensorflow Serving
6. Siraj's [repo for deploying Tensorflow to production](https://github.com/llSourcell/How-to-Deploy-a-Tensorflow-Model-in-Production)
7. https://www.tensorflow.org/versions/r1.2/programmers_guide/saved_model_cli
8. https://stackoverflow.com/questions/44125403/how-to-access-tensor-content-values-in-tensorproto-in-tensorflow
