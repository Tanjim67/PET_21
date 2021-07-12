# How to run the code
----

## Model Inversion
---
1. Clone the repo using one of these links: [HTTPS](https://github.com/kfaRabi/PET_21.git) or [SSH](git@github.com:kfaRabi/PET_21.git)
2. Change the working directory to the `model_inversion` folder (`cd model_inversion`).
3. Create a new python virtual environment using anaconda/miniconda (ex: `conda create --name pet_model_inversion python=3.8`)
4. Activate the newly created env. (`conda activate pet_model_inversion`)
5. Install `pip`. (`conda install pip`)
6. Install all the required python packages from the `requirements.txt` file using `pip`. (`pip install -r requirements.txt`)
7. Run the `model_inversion.py` file. (`python model_inversion.py`)


The last step will run the script. The script will first load all the face images from the `faces` folder and randomly split it into training (9 out of 10 images) and testing (1 out of 10 images) dataset. It will use the training images to train a 3 layer CNN model. At the end of the training, the training loss plot will be stored as an image file called `loss.png`. Then the model will be used to test the remaining images. After that, it will try to invert some of the faces from their label from the trained model using the MI-Face algorithm. This step will create a folder called `inverted_faces`, which will have the generated inverted faces and their loss.
