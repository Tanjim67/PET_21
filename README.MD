# How to run the code
----

## Creating Environment
---
1. Clone the repo using one of these links: [HTTPS](https://github.com/kfaRabi/PET_21.git) or [SSH](git@github.com:kfaRabi/PET_21.git) (`git@github.com:kfaRabi/PET_21.git`)
2. Change the working directory to the `PET_21` folder (`cd PET_21`).
3. Create a new python virtual environment using anaconda/miniconda (ex: `conda create --name pet python=3.8`)
4. Activate the newly created env. (`conda activate pet`)
5. Install `pip`. (`conda install pip`)
6. Install all the required python packages from the `requirements.txt` file using `pip`. (`pip install -r requirements.txt`)

## Model Inversion
Run the following command to run the model inversion attack:
```sh
python3 main.py -MINV -MINV-target="model_inversion/models/mi_target_model.pt"
```
Here, `-MINV` arg tells the `main.py` file to run the model inversion attack. and the `-MINV-target` is an optional argument to specify the path of an already trained target model. If the target model is passed, then the attack will not train a new target model.

Without specifying the `-MINV-target`, one can simply execute the following command:
```sh
python3 main.py -MINV
```

Since the AT&T face dataset has a specific structure, we have written a special file loader python class to evenly distribute each face class. Because of the special structure, we are not receiving any `datapath` arg for this attack.


The command will run the model inversion attack. The script will first load all the face images from the `faces` folder and randomly split it into training (9 out of 10 images) and testing (1 out of 10 images) dataset. It will use the training images to train a 3 layer CNN model. At the end of the training, the training loss plot will be stored as an image file called `loss.png`. Then the model will be used to test the remaining images. After that, it will try to invert some of the faces from their label from the trained model using the MI-Face algorithm. This step will create a folder called `inverted_faces`, which will have the generated inverted faces and their loss. If retrained, the model will be stored in a subfolder called `models`. If the target model path is specified the training steps will be skipped.

## Attribte Inference
---
Run the following command to run the attribute inference attack:
```sh
python3 main.py -AI -AI-dataset=UTKFacer -AI-target="attribute_inference/models/ai_target_model.pt" -AI-attack="attribute_inference/models/ai_attack_model.pt"
```

Here, `-AI` arg tells the `main.py` file to run the attribute inference attack. The remaining args and their meaning are listed below:
| arg name | meaning | is required |
|---|---|---|
|-AI-dataset|training and testing data path|true|
|-AI-target|path of a already trained target model|false (will train the target model if not given)|
|-AI-attack|path of a already trained attack model|false (will train the attack model if not given)|

To train a target and attack model from the beginning the following command can be used:
```sh
python3 main.py -AI -AI-dataset=UTKFacer
```

This will run the attribute inference attack. The script will first load all the face images from the `UTKFace` folder and randomly split it into training and testing dataset (80% train, 20% test). It will use the training images to train a 2 layer CNN model to predict the race. At the end of the training, the training loss plot will be stored as an image file called `target_train_loss.png`. Then the model will be used to test the remaining images. After that, the script will trian the attack model using the same training dataset. We first feed the images to the target model and get the embedding from the last fully connected layer which is then used to train the attack model where our labels are now gender. At the end of the attack model training an image containing the graph of the attack model loss function will be stored as `attack_train_loss.png`. All the train and test accuracy will be printed on the terminal.

If retrained, the model(s) will be stored in a subfolder called `models`. If the target model path is specified the training steps will be skipped. If the attack model path is specified the attack model trainiing steps will be skipped.


## main.py Usage

### membership_inference
```sh
python3 main.py -MIA -MIA-target="./target.pth" -MIA-train="PTH_TO_TARGET_TRAIN" -MIA-test="PTH_TO_TARGET_TEST" -MIA-s-train="PTH_TO_SHADOW_TRAIN" -MIA-s-test="PTH_TO_SHADOW_TEST"
```

### model_inversion
```sh
python3 main.py -MINV -MINV-dataset="model_inversion/faces/"
```
