# check_data.py
import os

# Paths to your data folders
fake_folder = 'data/training_fake'
real_folder = 'data/training_real'

# Count the number of images in each folder
num_fake = len(os.listdir(fake_folder))
num_real = len(os.listdir(real_folder))

print(f"Number of FAKE images: {num_fake}")
print(f"Number of REAL images: {num_real}")
print(f"Total images: {num_fake + num_real}")
print(f"Percentage of REAL images: {(num_real / (num_fake + num_real)) * 100:.2f}%")