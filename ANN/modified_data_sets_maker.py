import csv
import numpy as np
from numpy import save
import cv2
import random

with open("./data/train_set/raw/Y_train.csv", newline='') as f:
    reader = csv.reader(f)
    data = list(reader)


pokemon_not_grass_list = []
k = 0
counter = 0
while True:
    if data[k][1] != "Grass":
        data[k][1] = 0
        pokemon_not_grass_list.append(data[k])
        counter += 1
    if counter == 72:
        break
    k += 1

pokemon_grass_list = []
for i in range(len(data)):
    if data[i][1] == "Grass":
        data[i][1] = 1
        pokemon_grass_list.append(data[i])

train_list = pokemon_not_grass_list + pokemon_grass_list
random.shuffle(train_list)

X_train_modified = []
for k in range(len(train_list)):
    image = cv2.imread("./data/train_set/raw/" + train_list[k][0] + ".png")
    X_train_modified.append(image)

X_train_modified = np.array(X_train_modified)
X_train_modified = X_train_modified.reshape(144, -1).T
X_train_modified = X_train_modified.astype("float64")
X_train_modified = X_train_modified / 255
save("X_train_modified_preprocessed.npy", X_train_modified)


Y_train = []
for k in range(len(train_list)):
    Y_train.append(train_list[k][1])

print(Y_train)
Y_train = np.array(Y_train)
Y_train = Y_train.reshape(1, len(Y_train))
Y_train = Y_train.astype("float64")
print(Y_train.shape)
save("Y_train_modified.npy", Y_train)