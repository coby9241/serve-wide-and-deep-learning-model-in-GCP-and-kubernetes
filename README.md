# Serving a Wide & Deep Learning model in Google Cloud Platform and Kubernetes

This guide consists of two main parts:
1. Creating and training a wide & deep learning model, and exporting it
2. Containerizing the model and serving it locally
3. Serving it on the cloud

Throughout this guide, I highly recommend you follow these 2 links:
- Vitaly Bezgachev's awesome posts, [Part 1](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-1-make-your-model-ready-for-serving-776a14ec3198), [Part 2](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-2-containerize-it-db0ad7ca35a7) and [Part 3](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-3-into-the-cloud-7115ff774bb6)
- [Tensorflow's Inception Serving guide](https://www.tensorflow.org/serving/serving_inception)

as I myself followed them very closely when deploying my own model. This document is written with the intent of only reminding myself how to do the tensorflow serving portion so it may not be very helpful as a step by step guide, but Windows OS users may find this useful as well as people who intends to use GCP for deploying to production.

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

### Introduction

Much of the concept of a wide & deep learning model is explained in the original paper [here](https://www.google.com.sg/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjtpqS7ntzXAhUIR48KHQ6yBkwQFggnMAA&url=https%3A%2F%2Farxiv.org%2Fabs%2F1606.07792&usg=AOvVaw0Ou4rDtQZAXCC1QqxJPBIu) as well in the guide [here](https://www.tensorflow.org/tutorials/wide_and_deep), so I won't go into the specifics of it. Some of the stuff you may want to read up on your own will be 
- creating the wide and deep columns
- creating the cross columns
- creating the columns with embedddings
- training and evaluating the model

This model used in this case is highly referenced from yufengg's [model](https://github.com/yufengg/widendeep), although I updated the APIs for the features according to the ones used in the wide & deep learning guide by Tensorflow. 

This example uses [Kaggle Criteo's Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge) for training the recommendation model and it outputs a prediction of 1 or 0 whether an ad will be clicked or not. The dataset consists of two files, train.csv and eval.csv. The dataset consists of a ground truth label, 26 categorical variables and 13 interval variables.

### Running the model

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

## 2. Containerizing the model and serving it locally

### Install Docker/Docker Toolbox

As a side track, before you start on this part, remember to install Docker/Docker Toolbox(if you do not have Hyper-V).
For Docker Toolbox users, especially Windows users, the docker commands are not readly available in the command line. Here are some simple steps to enable it.

1. Run this command in your CLI
```
docker-machine env <name-of-your-docker-image> 
```
For mine it was 
```
docker-machine env default
```
which I checked from my Oracle VirtualBox.

2. These lines will then appear:
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

Back to the main topic, to create the image for Tensorflow serving, just clone the repository. You can clone it anywhere you like in Windows as long as you remember where it is, using this command:
```
git clone --recurse-submodules https://github.com/tensorflow/serving.git
```
Please remember to include the --recursive-submodules command. This will include Tensorflow as well as its models submodules during the cloning.

### Building the Docker Image

The building of the Docker Image is done using a Dockerfile that can be found in the *serving/tensorflow_serving/tools/docker* directory from the folder you just cloned. There are two types of files:
- Dockerfile.devel
- Dockerfile.devel-gpu (for GPU support)

For this case, due to money issues, I will be using the normal Dockerfile without GPU support.

We use this file to create the docker image. First, cd into the cloned *serving* folder.
```
cd serving
```
Then run the below command to build the docker image.
```
docker build --pull -t <username>/tensorflow-serving-devel -f tensorflow_serving/tools/docker/Dockerfile.devel .
```
Now run a container with the image you built:
```
docker run --name=tensorflow_container -it <username>/tensorflow-serving-devel
```
If successful, you should be inside the shell of the new docker image you created.

If you exit the shell by any means, to re-enter just run:
```
docker start -i tensorflow_container
```
to re-enter the shell.

Inside the shell of the container, make sure you are at the root directory and clone the Tensorflow serving repo:
```
cd ~
cd /
git clone --recurse-submodules https://github.com/tensorflow/serving.git
```
Next we configure our build.
```
cd /serving/tensorflow
./configure
```
Keep pressing enter without inputting in everything to accept the default. For this example, we can accept all defaults.
Before we move to the next step of building the Tensorflow serving, ensure that your Docker VM has enough RAM. One issue that may occur would be that the build terminate midway if it does not have enough RAM. To allocate more RAM in Virtualbox, stop the docker machine image, go to settings and give more RAM, and restart the machine again.

Now that you have enough RAM for the machine, you can start the build process by running this command. Make sure you are in the serving folder.
```
bazel build -c opt tensorflow_serving/...
```
This will take very long. For me it took ~6000 seconds.
Check if you are able to run the tensorflow_model_server using this command:
```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```
If yes, you are good to go, else if you encounter an error such as no such file or directory (like me) you have to install the ```tensorflow_model_server``` in your image externally using the instructions in this [link](https://www.tensorflow.org/serving/setup)

Note: this will affect your deployment of the image in the cloud later as the .yaml file used to deploy the Kubernetes Cluster will be affected by this. My guide later assumes the above command cannot work and I installed the ```tensorflow_model_server``` using ```apt-get```. If you wish to still use the above command to run it, I suggest you learn abit about how kubectl commands and the yaml file works to write your own file so that you know how to modify it. 

To install ```tensorflow_model_server```, follow these two steps:
1. Add TensorFlow Serving distribution URI as a package source (one time setup)
```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
```
2. Install and update TensorFlow ModelServer
```
apt-get update && apt-get install tensorflow-model-server
```
Once installed, the binary can be invoked using the command ```tensorflow_model_server```.

### Deploying the model locally

Next you copy the exported model with the protobuff and the variables folder. From the Windows CLI (not the docker shell):
```
cd <path before export folder>
docker cp ./export tensorflow_container:/serving
```
This copies the export folder to the /serving folder in the Docker Container tensorflow_container.

To check if the folder is copied properly.
```
cd /serving/export/1511606217
ls
```
You should see the saved_model protobuff and the variables folder.
```
root@1726471e9694:/serving/export/1511606217# ls
saved_model.pb  variables
```
To start hosting the model locally, run this command from the root directory ```/```:
```
root@1726471e9694:/# tensorflow_model_server --port=9000 --model_name=wide_deep --model_base_path=/serving/export &> wide_deep_log &
[2] 1600
```
Some explaination for the arguments. ```--port=9000``` specifies the port number and ```--model_name=wide_deep``` specifies the name. Both can be any number or any string but it is important as later in client python file these arguments will have to be added in when sending the request.
The ```model_base_path``` argument is fixed and it points to the directory of the export folder you copied just now to the docker container. ```&> wide_deep_log``` redirects the stdout to this file wide_deep_log. (see [this](https://superuser.com/questions/335396/what-is-the-difference-between-and-in-bash/335415)). The last ```&``` just executes this entire process in the background instead (if not it will be stuck there, try omit it and you will know what I mean).

To check if the model is hosted properly, run this command:
```
cat wide_deep_log
```
You should see output similar to this (especially the last line where it says running model server at host and port):
```
root@1726471e9694:/# cat wide_deep_log
2017-11-26 11:34:03.694919: I tensorflow_serving/model_servers/main.cc:147] Building single TensorFlow model file config:  model_name: wide_deep model_base_path: /serving/export
2017-11-26 11:34:03.695029: I tensorflow_serving/model_servers/server_core.cc:441] Adding/updating models.
2017-11-26 11:34:03.695043: I tensorflow_serving/model_servers/server_core.cc:492]  (Re-)adding model: wide_deep
2017-11-26 11:34:03.695428: I tensorflow_serving/core/basic_manager.cc:705] Successfully reserved resources to load servable {name: wide_deep version: 1511606217}
2017-11-26 11:34:03.695441: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: wide_deep version: 1511606217}
2017-11-26 11:34:03.695449: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: wide_deep version: 1511606217}
2017-11-26 11:34:03.695463: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:360] Attempting to load native SavedModelBundle in bundle-shim from: /serving/export/1511606217
2017-11-26 11:34:03.695473: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:236] Loading SavedModel from: /serving/export/1511606217
2017-11-26 11:34:03.747962: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:155] Restoring SavedModel bundle.
2017-11-26 11:34:03.781389: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:190] Running LegacyInitOp on SavedModel bundle.
2017-11-26 11:34:03.799166: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:284] Loading SavedModel: success. Took 103168 microseconds.
2017-11-26 11:34:03.801735: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: wide_deep version: 1511606217}
E1126 11:34:03.802407393    1600 ev_epoll1_linux.c:1051]     grpc epoll fd: 3
2017-11-26 11:34:03.803215: I tensorflow_serving/model_servers/main.cc:288] Running ModelServer at 0.0.0.0:9000 ...
```
Congrats, you have successfully served a Tensorflow model locally.

## 3. Serving it on the cloud

For me I used GCP, they have the $300 credits thing that you can use to try out these project. Just make sure you have the 300 dollar credits in your account under the billing section.

### Creating a project, installing Google SDK and Kubectl

Create a new project named tensorflow-serving. Remember this ID somewhere because it will be used later.
![New Project](/images/tensorflow_new_project.png)

Make sure you enabled the Container Engine API by going to the Kubernetes Window in the Project Dashboard.
![Enable API](/images/enable_container_API.png)

Install Google SDK as well.
![Install SDK](/images/install_SDK.png)

Finally, run this command in CLI to install Kubectl, command line interface for Kubernetes:
```
gcloud components install kubectl
```

### Preparing the container to push to cloud

First we need to find the container we need to push to the cloud. Run this command:
```
docker ps --all
```
This will output the relevant info of the Docker Containers. Take note of the CONTAINER ID column
```
CONTAINER ID        IMAGE                              COMMAND             CREATED             STATUS              PORTS               NAMES
1726471e9694        looboon/tensorflow-serving-devel   "/bin/bash"         21 hours ago        Up 12 hours                             tensorflow_container
```
Next, we create a docker image to push to GCP from the container using the CONTAINER ID.
```
docker commit 1726471e9694 looboon/tensorflow-serving-wide-deep:v1.0
```

Login to your GCP Project 
```
gcloud auth login --project tensorflow-serving
```
You will get a pop up to agree to something using your account, just press agree. Now you create a new Google Container Engine cluster for service deployment. For this example, two nodes will be fine.
```
gcloud container clusters create wide-deep-serving-cluster --num-nodes 2
```
Set the default cluster for gcloud container command and pass cluster credentials to kubectl.
```
gcloud config set container/cluster wide-deep-serving-cluster
gcloud container clusters get-credentials wide-deep-serving-cluster
```

### Upload the Docker image
Let's now push our image to the Google Container Registry so that we can run it on Google Cloud Platform.
First we list the images in CLI.
```
docker image ls
```
Results:
```
REPOSITORY                             TAG                 IMAGE ID            CREATED             SIZE
looboon/tensorflow-serving-wide-deep   v1.0                3f5e6138cb01        4 minutes ago       4.41GB
looboon/tensorflow-serving-devel       latest              46bb17334e0b        13 hours ago        1.07GB
ubuntu                                 16.04               20c44cd7596f        8 days ago          123MB
```
Tag the looboon/tensorflow-serving-wide-deep image using the Container Registry format and our project name. Take note, for the last argument, use your own project id the one with the numbers. That was you needed to remember your project ID. This instruction was missed out in the inception serving guide in the Tensorflow documentation.
```
docker tag looboon/tensorflow-serving-wide-deep:v1.0 gcr.io/<project-id>/wide_deep
```
For mine it will be:
```
docker tag looboon/tensorflow-serving-wide-deep:v1.0 gcr.io/tensorflow-serving-187205/wide_deep
```
Next we push the image to the Registry. Remember to replace your string with your own project id.
```
gcloud docker -- push gcr.io/tensorflow-serving-187205/wide_deep
```

### Create Kubernetes Deployment and Service

The deployment consists of 2 replicas of wide and deep inference server controlled by a Kubernetes Deployment. The replicas are exposed externally by a Kubernetes Service along with an External Load Balancer.

We create them using the Kubernetes config wide_deep_k8s.yaml that is in this repo. In this file,  a deployment controller and a service is defined. It is possible to deploy the entire cluster using kubectl, but the yaml file makes it more convinient. 

Warning though, due to some issues, you will need to modify this file for your own usage. I will explain abit. Some of the important things to note in the yaml config file:

The config file is split into two parts
1. Deployment
2. Service

#### Deployment
I want to deploy my Docker image into 2 Pods. This two lines specify that only two pods will be used:
```
spec:
  replicas: 2
```
and to pull them from my Google Cloud Registry as per below
```
spec:
  containers:
  - name: wide-deep-container
    image: gcr.io/tensorflow-serving-187205/wide_deep
```
This few lines above are very important, especially the image. This line must be the same as the image you pushed into the gcloud using the ```gcloud docker --push``` command you used previously. Change the project ID to your own.

After deployment a Pod should start the Shell and start TensorFlow, serving a wide-and-deep model, in the Docker container on the port 9000:
```
command:
- /bin/sh
- -c
args:
- tensorflow_model_server --port=9000 --model_name=wide_deep --model_base_path=/serving/export
ports:
- containerPort: 9000
```
Here is another important point. Remember when I said if you can run the tensorflow_model_server using ```bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server``` command, it might affect the deployment of the image as .yaml file used to deploy the Kubernetes Cluster will be affected by this? Well here it is. 

If you managed to run the tensorflow_model_server using ```bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server``` and did not apt-get install tensorflow_model_server, you just need to change the above line to ```/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=wide_deep --model_base_path=/serving/export``` and it will work fine as well.

#### Service
Service must accept external requests on port 9000 and forward them to a container port 9000 in a Pod:
```
ports:
- port: 9000
  targetPort: 9000
```  
And provide load balancing between 2 underlying Pods:
```
type: LoadBalancer
```

Now all we have to do is to deploy the Kubernetes service using the .yaml file
```
cd <yaml file location>
kubectl create -f wide_deep_k8s.yaml

deployment "wide-deep-deployment" created
service "wide-deep-service" created
```
Check the deployment:
```
kubectl get deployment

NAME                   DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
wide-deep-deployment   2         2         2            2           37s
```
Check the service:
```
kubectl get service

NAME                TYPE           CLUSTER-IP      EXTERNAL-IP      PORT(S)          AGE
kubernetes          ClusterIP      10.19.240.1     <none>           443/TCP          1m
wide-deep-service   LoadBalancer   10.19.247.108   35.197.136.221   9000:32314/TCP   50s
```
Check the pods:
```
kubectl get pods

NAME                                    READY     STATUS              RESTARTS   AGE
wide-deep-deployment-1890294076-d4lpz   0/1       ContainerCreating   0          19s
wide-deep-deployment-1890294076-sxngn   0/1       ContainerCreating   0          19s
```
Hint: Experience failing multiple times previously tells me that the pods are the most problematic ones and needs the debugging the most. For instance you may encounter an error like below where status is not ContainerCreating or Running (the correct ones) but instead something else:
```
kubectl get pods

NAME                                    READY     STATUS             RESTARTS   AGE
wide-deep-deployment-1578830529-5wwwj   0/1       ImagePullBackOff   0          1m
wide-deep-deployment-1578830529-l6bpz   0/1       ImagePullBackOff   0          1m
```

One useful tip to debug is check its logs. For instance, to check the details for pod ```wide-deep-deployment-1890294076-d4lpz```, we use this command:
```
kubectl describe pods wide-deep-deployment-1890294076-d4lpz
```
This will show some events that you can use to debug like the ones below:
```
Events:
  Type     Reason                 Age               From                                                          Message
  ----     ------                 ----              ----                                                          -------
  Normal   Scheduled              2m                default-scheduler                                             Successfully assigned wide-deep-deployment-1578830529-5wwwj to gke-wide-deep-serving-cl-default-pool-ab264915-p3xk
  Normal   SuccessfulMountVolume  2m                kubelet, gke-wide-deep-serving-cl-default-pool-ab264915-p3xk  MountVolume.SetUp succeeded for volume "default-token-t2wjz"
  Normal   Pulling                55s (x4 over 2m)  kubelet, gke-wide-deep-serving-cl-default-pool-ab264915-p3xk  pulling image "gcr.io/tensorflow-serving-187205/wide"
  Warning  Failed                 54s (x4 over 2m)  kubelet, gke-wide-deep-serving-cl-default-pool-ab264915-p3xk  Failed to pull image "gcr.io/tensorflow-serving-187205/wide": rpc error: code = 2 desc = Error: Status 405 trying to pull repository tensorflow-serving-187205/wide: "v1 Registry API is disabled. If you are not explicitly using the v1 Registry API, it is possible your v2 image could not be found. Verify that your image is available, or retry with `dockerd --disable-legacy-registry`. See https://cloud.google.com/container-registry/docs/support/deprecation-notices"
  Warning  FailedSync             7s (x12 over 2m)  kubelet, gke-wide-deep-serving-cl-default-pool-ab264915-p3xk  Error syncing pod
  Normal   BackOff                7s (x8 over 2m)   kubelet, gke-wide-deep-serving-cl-default-pool-ab264915-p3xk  Back-off pulling image "gcr.io/tensorflow-serving-187205/wide"
```
For this example, my image argument in the yaml file was wrong, i pushed the image into the Google Cloud Registry as ```gcr.io/tensorflow-serving-187205/wide_deep``` but instead i used ```gcr.io/tensorflow-serving-187205/wide``` for the image argument in the yaml file.

You can describe the service as well using the command below:
```
kubectl describe service wide-deep-service

Name:                     wide-deep-service
Namespace:                default
Labels:                   run=wide-deep-service
Annotations:              <none>
Selector:                 app=wide-deep-server
Type:                     LoadBalancer
IP:                       10.19.247.108
LoadBalancer Ingress:     35.197.136.221
Port:                     <unset>  9000/TCP
TargetPort:               9000/TCP
NodePort:                 <unset>  32314/TCP
Endpoints:                10.16.0.15:9000,10.16.1.16:9000
Session Affinity:         None
External Traffic Policy:  Cluster
Events:
  Type    Reason                Age   From                Message
  ----    ------                ----  ----                -------
  Normal  CreatingLoadBalancer  1m    service-controller  Creating load balancer
  Normal  CreatedLoadBalancer   24s   service-controller  Created load balancer
```

Once you reach this step, congrats, you have deployed the Kubernetes cluster successfully. All that is left is really to test the service with a client.

### Testing the service with a client

I have included a client file wide_and_deep_client.py. Inside the client file I have hardcoded a single data point as an example due to laziness (my bad), but it should not be difficult for you to do some argparse and accept a json file for instance and convert it into inputs for the model. To modify the inputs, just simply change my hard coded inputs for each input line into your own ones. For example, for input 'C1', if you have a variable ```input_C1``` extracted from some json file and etc, just change the line from:
```
request.inputs['C1'].CopyFrom(tf.contrib.util.make_tensor_proto("68fd1e64", shape=[1]))
```
to 
```
request.inputs['C1'].CopyFrom(tf.contrib.util.make_tensor_proto(input_C1, shape=[1]))
```
should do the trick.

Follow these steps to run the client py file.
```
cd <folder where client.py file is>
python wide_and_deep_client.py --server=35.197.136.221:9000
```
In this case ```35.197.136.221``` was my external IP which the service was hosted. It should be different for your case so do check it out using ```kubectl get service``` command.

You should see:
```
True label: 1
Prediction: 1
```
as the output. 
Uncomment this line:
```
# print('Prediction: ' + str(prediction))
``` 
if you want to see the entire TensorProto which will look like this:
```
Prediction: {u'probabilities': dtype: DT_FLOAT
tensor_shape {
  dim {
    size: 1
  }
  dim {
    size: 2
  }
}
float_val: 0.37165120244
float_val: 0.628348827362
, u'classes': dtype: DT_STRING
tensor_shape {
  dim {
    size: 1
  }
  dim {
    size: 2
  }
}
string_val: "0"
string_val: "1"
}
```
Congratulations if you reach this point, you would have deployed a wide and deep learning model in the cloud in Kubernetes.

## Conclusion

This entire document outlines the flow from creating a wide and deep learning model, to containerizing it, to finally deploying it on GCP in Kubernetes. This doucmentation is mainly for me to note down what I was doing but if you ever stumble upon it and find it useful it would be great, as inside it contains info I wish I knew when I first tried it out. I had to spend hours googling before I found out these stuff.

Then again, if these set of instructions don't work for you, don't give up. It is understandable as my setup will be different from yours (Ubuntu vs Windows, Python 2.7 vs 3.5 etc.). However, my advice will be to google any error message you see and likely there will be some way online you can use to solve your problem, because thats what I did for mine.

Please feel free to contact me if there's any questions or if there's anything wrong in this documentation.

## Credits and Useful Links (I'm spamming abit but that's how many links I referenced):

1. Vitaly Bezgachev's awesome posts, [Part 1](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-1-make-your-model-ready-for-serving-776a14ec3198), [Part 2](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-2-containerize-it-db0ad7ca35a7) and [Part 3](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-3-into-the-cloud-7115ff774bb6). He used Azure but I had some issues with Azure cli so I used GCP instead.
2. [Tensorflow's Inception Serving guide](https://www.tensorflow.org/serving/serving_inception)
3. Yufengg's [widendeep repo](https://github.com/yufengg/widendeep) which I got much of the wide and deep code from
4. Tensorflow's [Wide and Deep Tutorial](https://www.tensorflow.org/tutorials/wide_and_deep)
5. Weimin Wang's [blog post](https://weiminwang.blog/2017/09/12/introductory-guide-to-tensorflow-serving/) for Tensorflow Serving
6. Siraj's [repo for deploying Tensorflow to production](https://github.com/llSourcell/How-to-Deploy-a-Tensorflow-Model-in-Production)
7. https://www.tensorflow.org/versions/r1.2/programmers_guide/saved_model_cli
8. https://stackoverflow.com/questions/44125403/how-to-access-tensor-content-values-in-tensorproto-in-tensorflow
