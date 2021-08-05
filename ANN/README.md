# Pokemon ML Algorithm, Grass Type or Not (ANN)
This project implements a 2 layer, 4 hidden unit Artifical Neural Network on a dataset of 809 images of gen 1-7 Pokemon, learning if images of Pokemon are
grass type or not (1 or 0, respectively). The neural network is completely vectorized and was written as an L-layer Deep Neural Network, allowing for easy hyperparameter optimization.

## Implementation
### Technologies
- Python
- Numpy
- CV2
- Matplotlib

### Method
1. Imported the datasets that were preprocessed and vectorized in the logistic regression project *(see ../logistic regression)* 
2. Created a new dataset with a 50/50 split of grass and not grass-type Pokemon to verify the model's accuracy. This was also preprocessed and made into a numpy
array of shape (120 * 120 * 3, m) *note 120 * 120 * 3 is the amount of pixels across RGB of each image of the dataset*
3. Wrote the architecture of a L-layer ANN by creating a collection of helper functions
    - Parameter Initialization
      - Wrote a function that would would return randomly generated weights, with the right dimensions, and vectors of zeros for the biases for any given list of dimensions of the neural network
    - Forward Propagation
      - Wrote a function to calculate the linear component of each layer
      - Wrote a function to calculate the activation component of each layer
      - Combined the 2 functions above into a loop that does forward propagation over all L layers. Each iteration of the loop would also store that iteration's data in a list to be used in backward propagation
    - Backward Propagation
      - Wrote a function to calculate the activation component of backward propagation of each layer (using the list of forward propagation data)
      - Wrote a function to calculate the linear component of backward propagation of each layer (using the list of forward propagation data)
      - Combined the 2 functions above that does backward propagation over all L layers. Each iteration of the loop would also store the computed derivatives of each layer of backward propagation, to be used in gradient descent later
    - Parameter Optimization
      - Wrote a function that updated the parameters using the derivatives calculated in backward propagation
    - Cost Calculation
      - Calculated the cost of each iteration by taking the average of the losses calculated (each iteration processes the entire training set at a time), and the loss function that was used was binary cross-entropy
4. After experimenting with different hyperparameters, I found that 1 hidden layer with 4 units and a learning rate of 0.1 produced the optimum results
     
## Results
The model, after 2000 iterations, had a train accuracy of 100%, a test accuracy of 86%, and a test accuracy on the dataset which is 50/50 grass-type of 100%.


## Learning Rate
![image](https://user-images.githubusercontent.com/83722101/128434928-af4f5ce4-a843-40f6-9772-b5be30d5b4d4.png)


## Acknowledgements
The dataset used was scrapped by Vishal Subbiah, <a target="_blank" href="https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types?select=images">found here</a>
