import os
import shutil

source_dir = '/home/jupyter-warapob/Research/FaceDetection/widerface/WIDER_train/images'

# Iterate through each subfolder
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # Get the full path of the image file
        file_path = os.path.join(root, file)
        print(file_path,source_dir)
        # Move the image file to the target directory
        shutil.move(file_path, source_dir)