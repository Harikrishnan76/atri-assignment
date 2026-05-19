# atri-assignment

Problem Statement
In this assignment, you need to implement a feedforward neural network and write the backpropagation code for training the network. We strongly recommend using numpy for all matrix/vector operations. You are not allowed to use any automatic differentiation packages. This network will be trained and tested using the Fashion-MNIST dataset. Specifi cally, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes.

Question 1 (2 Marks)
Download the Fashion-MNIST dataset and plot 1 sample image for each class as shown in the grid below. Use "from keras. datasets import fashion_mnist" for getting the Fashion MNIST dataset.

Question 2 (10 Marks)
Implement a feedforward neural network that takes images from the Fashion-MNIST data as input and outputs a probability distribution over the 10 classes.
Your code should be fl exible, allowing for easy modifi cation of the number of hidden layers and the number of neurons in each hidden layer.
We will check the code for implementation and ease of use.

Question 3 (18 Marks)
Implement the backpropagation algorithm with support for the following optimisation functions
● sgd
● momentum-based gradient descent
● Nesterov accelerated gradient descent
● rmsprop
● adam
● nadam
(12 marks for the backpropagation framework and 2 marks for each of the optimisation algorithms above)
We will check the code for implementation and ease of use (e.g., how easy it is to add a new optimisation algorithm such as Eve).


Question 4 (10 Marks)
Use the sweep functionality provided by Wandb to fi nd the best values for the hyperparameters listed below. Use the standard train/test split of fashion_mnist (use (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()). Keep 10% of the training data aside as validation data for this hyperparameter search. Here are some suggestions for different values to try for hyperparameters. As you can quickly see that this leads to an exponential number of combinations. You will have to think about strategies to do this hyperparameter search effi ciently. Check out the options provided by wandb. Sweep and write down what strategy you chose and why.
● number of epochs: 5, 10
● number of hidden layers: 3, 4, 5
● size of every hidden layer: 32, 64, 128
● weight decay (L2 regularisation): 0, 0.0005, 0.5
● learning rate: 1e-3, 1 e-4
● optimizer: sgd, momentum, nesterov, rmsprop, adam, nadam
● batch size: 16, 32, 64
● weight initialisation: random, Xavier
● activation functions: sigmoid, tanh, ReLU

Wandb will automatically generate the following plots. Paste these plots below using the "Add Panel to Report" feature. Make sure you use meaningful names for each sweep (e.g., hl_3_bs_16_ac_tanh to indicate that there were 3 hidden layers, batch size was 16, and activation function was ReLU) instead of using the default names (whole-sweep, kind-sweep) given by wandb.

Question 5 (5 marks)
We would like to see the best accuracy on the validation set across all the models that you train.
Wandb automatically generates this plot, which summarises the test accuracy of all the models that you tested. Please paste this plot below using the "Add Panel to Report" feature

Question 6 (20 Marks)
Based on the different experiments that you have run, we want you to make some inferences about which confi gurations worked and which did not.
Here again, wandb automatically generates a "Parallel co-ordinates plot" and a "correlation summary" as shown below. Learn about a "Parallel co-ordinates plot" and how to read it.
By looking at the plots that you get, write down some interesting observations (simple bullet points, but should be insightful). You can also refer to the plot in Question 5 while writing these insights. For example, in the above sample plot, many confi gurations give less than 65% accuracy. I would like to zoom in on those and see what is happening.
I would also like to see a recommendation for what confi guration to use to get close to 95% accuracy.

Question 7 (10 Marks)
For the best model identifi ed above, report the accuracy on the test set of fashion_mnist and plot the confusion matrix as shown below. More marks for creativity (fewer marks for producing the plot shown below as it is)

Question 8 (5 Marks)
In all the models above, you would have used cross-entropy loss. Now compare the cross-entropy loss with the squared error loss. I would again like to see some automatically generated plots or your own plots to convince me whether one is better than the other.

Question 9 (10 Marks)
Paste a link to your GitHub code for this assignment
● We will check for coding style, clarity in using functions, and a README fi le with clear instructions on training and evaluating the model (the 10 marks will be based on this)
● We will also run a plagiarism check to ensure that the code is not copied (0 marks in the assignment if we fi nd that the code is plagiarized)
● We will also check if the training and test data have been split properly and randomly. You will get 0 marks on the assignment if we fi nd any cheating (e.g., adding test data to training data) to get a higher accuracy
Question 10 (10 Marks)
Based on your learnings above, give me 3 recommendations for what would work for the MNIST dataset (not Fashion-MNIST). Just to be clear, I am asking you to take your learnings based on extensive experimentation with one dataset and see if these learnings help on another dataset. If I give you a budget of running only 3 hyperparameter confi gurations as opposed to the large number of experiments you have run above, then which 3 would you use and why? Report the accuracies that you obtain using these 3 confi gurations.
