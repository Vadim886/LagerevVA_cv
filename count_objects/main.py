import numpy as np
import matplotlib.pyplot as plt

external = np.diag([1, 1, 1, 1]).reshape(4, 2, 2)

internal = np.logical_not(external)

cross = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])

def match(a, masks):
    for mask in masks:
        if np.all(a == mask):
            return True
    return False

def count_objects(image):
    E = 0
    for y in range(0, image.shape[0] - 1):
        for x in range(0, image.shape[1] - 1):
            sub = image[y : y + 2, x : x + 2]
            if match(sub, external):
                E += 1
            elif match(sub, internal):
                E -= 1
            elif match(sub, cross):
                E += 2
    return E / 4

img1 = np.load("./example1.npy")
img1[img1 != 0] = 1
print("example1.npy:", count_objects(img1))

img2 = np.load("./example2.npy")
total_objects = 0

for i in range(img2.shape[-1]): # Создаем копию канала и бинаризуем его
    channel = img2[:, :, i].copy()
    channel[channel != 0] = 1

    total_objects += count_objects(channel)

print("example2.npy:", total_objects)