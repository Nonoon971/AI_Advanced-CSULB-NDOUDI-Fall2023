# AI_Advanced-CSULB-NDOUDI-Fall2023
List of the assignment done during the Fall 2023 Artificial Intelligence Advanced class

### Assignment 1 - Optimizing Airport Locations Using Gradient Descent
In this assignment, the objective was to solve the n-airports problem, which involves determining the optimal locations for a set of airports to minimize the total distance to a set of cities. The solution was implemented using a gradient-based optimization algorithm in Python. 

The assignment required:
* Defining an objective function to calculate the distance between airports and cities.
* Applying gradient descent to iteratively adjust airport positions for minimizing the objective function.
* Visualizing the optimization process through a plot showing how the objective values change over time.

### Assignment 2 - Introduction to Neural Networks with Python: Classification and Regression
This assignment focuses on getting hands-on experience with fundamental deep learning concepts and neural network implementation using Python. 

It involved:
* Understanding the basics of neural networks.
* Reproducing three example programs that demonstrate different types of machine learning problems:
    * Binary Classification: Classifying movie reviews into positive or negative sentiments (bin_class.ipynb)
    * Multiclass Classification: Categorizing newswires into multiple classes (mul_class.ipynb).
    * Regression: Predicting house prices based on various features (reg.ipynb).

The assignment highlights key deep learning tasks including data preprocessing, model building, training, and evaluation in Python.

### Assignment 3 - Building a Multi-Layer Neural Network from Scratch in Python
In this assignment, the task was to implement a multi-layer neural network from the ground up, without using any external deep learning libraries like Keras, TensorFlow, or PyTorch. The key objectives included:

* Designing a neural network with two layers, where:
    * Layer 1 has a width of 2 and uses hyperbolic tangent as the activation function.
    * Layer 2 has a width of 1 and uses a sigmoid activation function.
* Calculating gradients for weights and biases using backpropagation and implementing these in a Jupyter notebook with LaTeX equations.
* Optimizing the parameters through gradient descent and predicting outputs based on given inputs.
* Reporting training and validation losses to evaluate model performance.

This assignment emphasizes a foundational understanding of neural network mechanics, focusing on manual implementation of training algorithms and model evaluation.

### Assignment 4 - Exploring Neural Network Regularization Techniques with Keras
This assignment involved designing a neural network and exploring various regularization methods to address overfitting. The tasks included:

* Creating a complex neural network with high epochs, intentionally leading to overfitting to observe a highly non-linear decision boundary.
* Implementing and testing regularization methods such as:
    * L2 Regularization: To penalize large weights and help in reducing overfitting.
    * Dropout: To randomly drop units during training, which prevents over-reliance on any specific neurons.
* Visualizing decision boundaries for the baseline model, the model with L2 regularization, and the model with Dropout.
* Comparing the accuracy of these models to understand the effectiveness of each regularization method.

The assignment demonstrates the practical use of regularization techniques to improve model generalization using the Keras library in Python.

### Assignment 5 - Hyperparameter Tuning with GridSearchCV for Neural Networks
This assignment focused on optimizing hyperparameters for a neural network using GridSearchCV. The key tasks included:

* Using a toy dataset and network design similar to Assignment 3.
* Employing the Adam optimizer to optimize the model and determining the best values for its parameters: beta_1, beta_2, and learning_rate.
* Conducting a grid search with GridSearchCV, testing at least three values for each hyperparameter.
* Printing the 'negative mean squared error' and corresponding hyperparameters for each combination.
* Identifying and reporting the best hyperparameter combination that minimizes the error.

This assignment underscores the importance of hyperparameter tuning in machine learning to enhance model performance.





