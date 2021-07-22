# Pokemon ML Algorithm, Grass Type or Not 
This project implements a simple, one-layer neural network (logistic regression) on a dataset of 809 images of gen 1-7 Pokemon, learning if images of Pokemon are
grass type or not (1 or 0, respectively). The neural network is completly vectorized. With the number of iterations of forward & backward propagation being 5,000 and a learning rate of 0.001, train and
test accuracy is 96% and 88%, respectively.

## Implementation
### Technologies
- Python
- Numpy
- CV2
- Matplotlib

### Method
1. Edited the . csv files to have each element in the list contain ["pokemon name", {1/0}], 1/0 based on if grass type or not, rather than ["pokemon name", "type"]

2. Created a script that took the images in the train/test datasets and combined them into a (120 * 120 * 3, m) numpy aray, with each image as a column vector,
ordered by their relative place in the .csv file containing information regarding whether or not the Pokemon is a grass type. Each array was saved as a .npy file (X)

3. Created a script that converted the .csv files into a 1xm numpy array, made of only 1s and 0s. Each array was saved as a .npy file (Y)

4. Wrote the architecture of the NN by writing each function before calling model()
    - Initialized parameters w, b, an [nx x 1] column vector and real number respectively, with zeros. <em>note nx = (120 * 120 * 3 for this dataset)</em>
    - Forward Propagation
      - Made a row vector A (1xm) of the activations of the entire set, by Z =  w.T dot X + b and A = sigmoid(Z) *note b is broadcasted*
      - Used cost = <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{m}(\sum_{i=1}^{m} -(y^{(i)}ln(a^{(i)} + (1-y^{(i)})ln(1-a^{(i)})))">,
    vectorized by taking the sum of (ln(a) dot Y.T) + (ln(1-a) dot (1-Y).T) <em>note Y is a 1xm row vector of correct grass-type predictions, and that the sum of 
    dot products works because of the piecewise nature of elements of Y being 1 or 0</em>
    - Backward Propagation
      - Made a 1xm row vector dz for each <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial J}{\partial t}"> per training column vector in X
      by taking dz = A-Y *(that is what the partial derivative equals to)*
      - Made a nx x 1 column vector dw, containing the sum of all <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial J}{\partial w}">, by taking
      (1/m) * X dot dz.T *(note the partial derivative calculation is correct because for each <img src="https://render.githubusercontent.com/render/math?math=w^{(i)}">,
        <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L}{\partial z^{(m)}} \frac{\partial z^{(m)}}{\partial w^{(i)}} = \frac{\partial L}{\partial w^{(i)}}">, <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial z^{(m)}}{\partial w^{(i)}} = X^{(i)}_{m}">
        and cost is the average of the losses
      - Calculated db as (1/m) * the sum of dz, since b is a constant, so <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial b} = \frac{\partial L}{\partial z}">
    - Optimization
      - Implemented gradient descent on parameters w and b *(note all w1 through wnx are subtracted/added at the same time due to vectorization)*
5. Created model() by adding the functions listed above
6. Adjusted original learning rate used (0.25) to 0.001 so the learning curve would be more smooth

## Accuracy after 5000 iterations
<img width="243" alt="Capture" src="https://user-images.githubusercontent.com/83722101/126699100-6d4bf3e9-b74a-413f-a061-6e1ddc4bf8dc.PNG">

## Learning Rate
<img width="460" alt="Capture" src="https://user-images.githubusercontent.com/83722101/126698658-f5d68fd7-4e11-4a2c-b530-a8adfd007a25.PNG">

## Acknowledgements
The dataset used was scrapped by Vishal Subbiah, <a target="_blank" href="https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types?select=images">found here</a>
