# MembershipInferencePETS
Project repo for Attack 1 of the PETS lecture: Membership Inference Attack using pytorch

## Setup  

I use the pip install of pytorch on WSL, I havent managed to get the gpu support working  
so i could only test with a CPU device, although it should work with gpu.  
tl;dr: follow the official pytorch [installation](https://pytorch.org/get-started/locally/)

## Usage

Linux:  
```sh  
python3 MIA.py
```

## Project structure  
#### MIA.py  
contains the main and puts everything together.  
- pick names for the models you train  
- set retrain to True if you want to overwrite the older models  
- if you dont want to overwrite just change the names of the models you want to retrain.  

#### nn.py  
contains our neural network classes  
- Conv2DNet is a simple CNN used by the target and shadow models  
- AttackNet is a simple binary classification NN with 4 fully connected hidden layers of size 1024  
#### util.py  
contains showdata function and everything log level  
#### data.py  
contains all functions that are used to collect/process data(sets) into data_loaders.  
this includes the attack model dataset creation with the deltaing of shadow,target output.

### model/ directory  

#### attack.py  
contains the training function of the attack model train_attack()  
and attack_infer(loader, model_path) that is used to evaluate the accuracy of a given attack model.  
#### shadow.py  
contains the training function of the shadow model train_attack()  
#### target.py  
contains the training function of the target model train_attack()  

### *.pth
currently all trained models are saved to "model/name.pth", can be configured in MIA.py

## Function  
note: there are comments in MIA.py highlighting this as well  

First we train our heavily overfitted target model if we dont have one given,  
then we train our shadow model on a dataset having the same distribution like the target dataset. We do this by reusing the mnist dataset.  

Thirdly we need to train our attack model. For this we have to create a dataset by evaluating both the target and shadow model with again, the same distribution dataset, and then subtracting the posteriors from each other = deltaing.  
These deltas will be given a label based on whenever they were part of the training or testing dataset. since the target model hasnt seen the testing data we give it the label 0 for non-member and training data label 1 for member.  
Essentially this makes the attack model learn whenever the target model has been trained with an image or not which is the goal.  

At the end of the MIA.py file we test the accuracy of the attack model on our testing and training data of the MNIST dataset. One can expand this by creating their own dataloader in data.py and then using it in the attack_infer function to get the accuracy.  

The project only works with the MNIST dataset right now.  
for the CIFAR10 dataset we have to change either the model or the loader to compensate for the 3 channels vs 1.  



## todos:
pipenv?  
indepth tuning  

### in-line stuff:  
make the train/test dataset for the attack model lazy?  
pick a consistent naming convention for function names lmfao (WIP)  
- attack_infer() -> function exists in attack.py, all lower case, seperator='_'
