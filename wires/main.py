import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import (binary_closing,binary_opening,binary_dilation,binary_erosion)

data = np.load("./wires5npy.txt")

labeled = label(data) # Маркируем изображение

total_wires = np.max(labeled)
print(f"Количество проводов: {total_wires}") # Считаем количество объектов

result = binary_erosion(data, np.ones(3).reshape(3,1)) # Разделить на части
labeled_eroded = label(result)

for wire_num in range(1, total_wires + 1):
    original_wire = (labeled == wire_num)

    wire_parts = labeled_eroded.copy()
    wire_parts[~original_wire] = 0
    num_parts = np.max(label(wire_parts))

    holes = num_parts - 1 if num_parts > 0 else 0 # Количество дырок = количество частей - 1

    if holes > 0:
        print(f"Провод {wire_num}: порван, в нём {holes} дырок")
    else:
        print(f"Провод {wire_num}: целый: дырок нет")

plt.subplot(121)
plt.imshow(data)
plt.subplot(122)
plt.imshow(result)
plt.show()
