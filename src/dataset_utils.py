import os
import shutil
import random


def create_split(src_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Crée les dossiers train/val/test à partir du dossier raw
    """
    classes = os.listdir(src_dir)
    for cls in classes:
        cls_path = os.path.join(src_dir, cls)
        images = os.listdir(cls_path)
        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        for i, img in enumerate(images):
            src_img = os.path.join(cls_path, img)
            if i < n_train:
                shutil.copy(src_img, os.path.join(train_dir, cls, img))
            elif i < n_train + n_val:
                shutil.copy(src_img, os.path.join(val_dir, cls, img))
            else:
                shutil.copy(src_img, os.path.join(test_dir, cls, img))
