import sys
import os

HELP = """Manual Page
Options:
--help                          # Print this message

========================= Membership Inference Attack =========================
-MIA                            # Execute the Membership Inference Attack.
-MIA-target="PATH_TO_MODEL"     # Specify the target model.

# Note inculde either 0 or all 4 options from below at once.
-MIA-train="PATH_TO_DATASET"    # Specify path to target training dataset.
-MIA-test="PATH_TO_DATASET"     # Specify path to target testing dataset.
-MIA-s-train="PATH_TO_DATASET"  # Specify path to shadow model training dataset.
-MIA-s-test="PATH_TO_DATASET"   # Specify path to shadow model testing dataset.

Example: -MIA -MIA-target="target.pth" -MIA-train="" -MIA-test="" -MIA-s-train="" -MIA-s-test=""
=============================== Model Inversion ===============================
-MINV                             # Execute the Model Inversion Attack.
-MINV-target="PATH_TO_MODEL"      # Specify the target model.
-MINV-train="PATH_TO_DATASET"     # Specify path to the training dataset.
-MINV-test="PATH_TO_DATASET"      # Specify path to the testing dataset.

============================= Attribute Inference =============================
-AI                             # Execute the Attribute Inference Attack.
-AI-target="PATH_TO_MODEL"      # Specify the target model.
-AI-train="PATH_TO_DATASET"     # Specify path to the training dataset.
-AI-test="PATH_TO_DATASET"      # Specify path to the testing dataset.


for additional information refer to the readme or other help documents"""

# Helper
def parse_EQ(s):
    return "".join(s.split("=")[1:])

def starts_with(start, text):
    if len(start) > len(text):
        return False
    return start == text[:len(start)]

if "--help" in sys.argv:
    print(HELP)
    exit(0)


MIA  = False
MIA_ARG = []
MINV = False
MINV_ARG = []
AI   = False
AI_ARG = []

for arg in sys.argv:
    if starts_with("-MIA", arg):
        MIA = True
        MIA_ARG.append(arg)
    elif starts_with("-MINV", arg):
        MINV = True
        MINV_ARG.append(arg)
    elif starts_with("-AI", arg):
        AI = True
        AI_ARG.append(arg)

if MIA:
    target_model = ""
    train_dataset = ""
    test_dataset = ""
    s_train_dataset = ""
    s_test_dataset = ""
    for arg in MIA_ARG:
        if   starts_with("-MIA-target=", arg):
            target_model = parse_EQ(arg)
        elif starts_with("-MIA-train=", arg):
            train_dataset = parse_EQ(arg)
        elif starts_with("-MIA-test=", arg):
            test_dataset = parse_EQ(arg)
        elif starts_with("-MIA-s-train=", arg):
            s_train_dataset = parse_EQ(arg)
        elif starts_with("-MIA-s-test=", arg):
            s_test_dataset = parse_EQ(arg)
    # print(target_model, train_dataset, test_dataset, s_train_dataset, s_test_dataset)
    if train_dataset != "" or test_dataset != "" or s_train_dataset != "" or s_test_dataset != "":
        if train_dataset == "" or test_dataset == "" or s_train_dataset == "" or s_test_dataset == "":
            print("ERROR, Please specify ALL 4 dataset parts for MIA. \n")
            print(HELP)
        else:
            CMD = "python3 membership_inference/MIA.py {} {} {} {} {} ".format(train_dataset, test_dataset, s_train_dataset, s_test_dataset, target_model)
            print("Executing " + CMD)
            os.system(CMD)

if MINV:
    print(MINV_ARG)
if AI:
    print(AI_ARG)
