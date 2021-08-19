import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# others
import os
# local
from util import *
from nn import *
from data import *
from models.target import *
from models.shadow import *
from models.attack import *

warn("ignoring all warnings of category UserWarning.")

if __name__ == '__main__':

    target_path = "models/target50.pth"
    shadow_path = "models/shadow37.pth"
    attack_path = "models/attack0599.pth"
    debug("target_path: " + target_path)
    debug("shadow_path: " + shadow_path)
    debug("attack_path: " + attack_path)

    retrain = False
    debug("retraining: " + str(retrain))

    # we have to train our target model for an example, the goal is to overfit this
    if retrain or not os.path.isfile(target_path):
        debug("collecting dataset for target model training")
        train_loader_target, test_loader_target = get_mnist_loaders()
        debug("start training target model")
        train_target(train_loader_target, test_loader_target, target_path)

    # we have to train 1 shadowmodel, dont overfit
    if retrain or not os.path.isfile(shadow_path):
        debug("collecting dataset for shadow model training")
        train_loader_shadow, test_loader_shadow = get_mnist_loaders()
        debug("start training shadow model")
        train_shadow(train_loader_shadow, test_loader_shadow, shadow_path)

    # now we have to train our binary test nn, we feed our shadow,target model with img from
    # MNIST training dataset with label 1 since they are members
    # MNIST testing dataset with label 0 since they are non members
    # then we take the posteriors from the shadow,target model and calulate the delta between them.
    # the delta is then used as input for the attack model.

    if retrain or not os.path.isfile(attack_path):
        debug("collecting dataset for attack model training")
        member_DL, nonmember_DL = get_mnist_loaders()
        train_data_loader, test_data_loader = get_attack_data_loaders(member_DL, nonmember_DL, target_path, shadow_path)
        debug("start training attack model")
        train_attack(train_data_loader, test_data_loader, attack_path)

    info("All models are trained")
    info("generating overall statistics now:")
    debug("collecting dataset for attack model evaluation")
    member_DL, nonmember_DL = get_mnist_loaders()
    train_data_loader, test_data_loader = get_attack_data_loaders(member_DL, nonmember_DL, target_path, shadow_path)
    info("collected datasets")
    info("measuring accuracy on the Testing data")
    attack_infer(test_data_loader, attack_path)
    info("measuring accuracy on the Training data")
    attack_infer(train_data_loader, attack_path)
    info("done, quitting now")
