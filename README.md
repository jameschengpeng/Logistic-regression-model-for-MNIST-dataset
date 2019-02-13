# Logistic-regression-model-for-MNIST-dataset
This model is implemented from Scratch in Python without TensorFlow or PyTorch
The best test accuracy of my model is around 91%. I defined the indicator vector
generator function, softmax function, gradient function in advance. The gradient
function is G = -(e(y) – softmax(theta*X))^(T) – X^(T). The most difficult part is
setting the learning rate. I tried C0/(C1 + l) at first, but the accuracy was only
around 60%. Then I tried constant learning rate and found that 0.02 was
appropriate. Therefore, I used 0.02 as the learning rate and trained 90000
epochs.
