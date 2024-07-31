import os
import shutil
import random

image_dir = "dataset/images"
label_dir = "dataset/labels"


def main():
    os.makedirs(os.path.join(image_dir, "train"))
    os.makedirs(os.path.join(label_dir, "train"))
    os.makedirs(os.path.join(image_dir, "val"))
    os.makedirs(os.path.join(label_dir, "val"))

    image_files = [
        f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))
    ]
    random.shuffle(image_files)
    label_files = [f.replace(".png", ".txt") for f in image_files]

    split_index = int(len(image_files) * 0.8)

    train_images, val_images = image_files[:split_index], image_files[split_index:]

    for f in train_images:
        shutil.move(os.path.join(image_dir, f), os.path.join(image_dir, "train", f))
        shutil.move(
            os.path.join(label_dir, f.replace(".png", ".txt")),
            os.path.join(label_dir, "train", f.replace(".png", ".txt")),
        )

    for f in val_images:
        shutil.move(os.path.join(image_dir, f), os.path.join(image_dir, "val", f))
        shutil.move(
            os.path.join(label_dir, f.replace(".png", ".txt")),
            os.path.join(label_dir, "val", f.replace(".png", ".txt")),
        )
    return


if __name__ == "__main__":
    main()
