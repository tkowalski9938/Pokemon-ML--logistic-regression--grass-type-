import csv
import numpy as np
from numpy import save
import cv2

#if using, make sure to change the paths to the respective .csv files

# with open('Y_train.csv', newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)

# new_list = []
# for k in range(len(data)):
#     new_list.append(data[k][1])

# print(new_list)

# for i in range(len(new_list)):
#     if new_list[i] == "Grass":
#         new_list[i] = 1
#     else:
#         new_list[i] = 0
# print(new_list)
# new_list = np.array(new_list)
# print(new_list)
# new_list = new_list.reshape(1, len(new_list))
# print(new_list.shape)
# print(new_list)
# save("Y_train.npy", new_list)

# with open('Y_test.csv', newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)


# X_test = []
# for k in range(len(data)):
#     image = cv2.imread ("./datasets/test_set/" + data[k][0] + ".png")
#     X_test.append(image)

# X_test = np.array(X_test)
# print(X_test.shape)
# #add the transpose!
# X_test = X_test.reshape(50, -1).T
# print(X_test.shape)
# counter = 0
# badcounter = 0
# for i in range(len(X_test[0])):

#     if X_test[0][i] != 0:
#         print("yay!")
#         counter += 1
#     elif X_test[0][i] == 0:
#         print("NO!")
#         badcounter += 1
# print(str(counter) + " good")
# print(str(badcounter) + " bad")
# print(len(X_test[0]))


# X_test = X_test.astype("float64")
# X_test = X_test / 255
# save("X_test_preprocessed.npy", X_test)





# with open("Y_train.csv", newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)

# print(len(data))
# print(data[0])
# X_train = []
# for k in range(len(data)):
#     image = cv2.imread("./datasets/train_set/" + data[k][0] + ".png")
#     X_train.append(image)

# X_train = np.array(X_train)
# print(X_train.shape)
# X_train = X_train.reshape(759, -1).T
# print(X_train.shape)
# X_train = X_train.astype("float64")
# X_train = X_train / 255
# save("X_train_preprocessed.npy", X_train)