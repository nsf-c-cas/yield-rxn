import shutil
import os
import shutil

task_name = "su"
for i in range(10):
    root = task_name + "/" + str(i)
    # shutil.rmtree(root + "/train")
    # shutil.rmtree(root + "/test")
    os.makedirs(root + "/train", exist_ok=True)
    os.makedirs(root + "/test", exist_ok=True)
    os.makedirs(root + "/train/raw", exist_ok=True)
    os.makedirs(root + "/train/processed", exist_ok=True)
    os.makedirs(root + "/test/raw", exist_ok=True)
    os.makedirs(root + "/test/processed", exist_ok=True)

    src = root + "/train_set.csv"
    dst = root + "/train/raw"
    shutil.copy(src, dst)

    src = root + "/test_set.csv"
    dst = root + "/test/raw"
    shutil.copy(src, dst)

