import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2

X_train = np.load("./data/train_set/vectorized/X_train_preprocessed.npy")
X_test = np.load("./data/test_set/vectorized/X_test_preprocessed.npy")
Y_train = np.load("./data/train_set/vectorized/Y_train.npy")
Y_test = np.load("./data/test_set/vectorized/Y_test.npy")

def sigmoid(z):
    z = np.clip(z, -500, 500 )
    return 1 / (1 + np.exp(-z))

def initialize(n_x):
    w = np.zeros([n_x, 1])
    b = 0.0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]

    #forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    J = (-1/m) * (np.dot(np.log(A + 1e-8), Y.T) + np.dot(np.log(1-A + 1e-8), (1-Y).T))
    J = np.squeeze(np.array(J))

    #backward propagation
    dz = A - Y
    dw = (1/m) * np.dot(X, dz.T)
    db = (1/m) * np.sum(dz)

    derivs = {"dw": dw, "db": db}

    return derivs, J

def optimize(w, b, X, Y, num_iterations, alpha):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    cost_list = []

    for k in range(num_iterations):
        derivs, J = propagate(w, b, X, Y)
        dw = derivs["dw"]
        db = derivs["db"]
        
        w = w - alpha * dw
        b = b - alpha * db

        if k % 100 == 0:
            cost_list.append(J)
            print("Cost after iteration %i: %f" %(k, J))
    
    params = {"w": w, "b": b}

    return params, cost_list

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros([1, m])

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0][i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    
    return Y_prediction

def model(X_train, X_test, Y_train, Y_test, num_iterations, alpha):
    w, b = initialize(X_train.shape[0])
    params, costs_lists = optimize(w, b, X_train, Y_train, num_iterations, alpha)
    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    stats = {"costs list": costs_lists,
     "Y_prediction_test": Y_prediction_test, "Y_prediction_train": Y_prediction_train,
     "w": w, "b": b, "learning rate": alpha, "num_iterations": num_iterations}

    return stats


model = model(X_train, X_test, Y_train, Y_test, 5000, 0.001)

#test your own images, then see learning curve
while True:
    path = input("Enter the path to a image you want to test for grass type pokemon or not \nIf you are done with testing and would like to see the learning curve, enter [break] \n")
    if path == "break":
        break
    try:
        image = cv2.imread(path)
        image = cv2.resize(image, [120,120])
        image = np.array(image)
        image = image / 255
        image = image.reshape(1, 43200).T
        prediction = predict(model["w"], model["b"], image)

        if np.squeeze(prediction) == 1:
            print("The algorithm thinks this image is a grass-type Pokemon \n")
        else:
            print("The algorithm thinks this image is not a grass-type Pokemon \n")
        
    except Exception as e:
        print(e)
        print("not a valid path \n")

costs = np.squeeze(model["costs list"])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(model["learning rate"]))
plt.show()