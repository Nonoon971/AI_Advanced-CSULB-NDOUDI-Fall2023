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

### Assignment 6 - Implementing Convolutional Neural Networks with Keras
This assignment involved implementing convolutional neural networks (CNNs) and experimenting with regularization methods to improve model performance and prevent overfitting. The tasks included:

* Data Organization: Setting up training, validation, and test directories for the Cats vs. Dogs dataset.
* Model Building: Constructing a CNN with multiple Conv2D and MaxPooling2D layers, followed by a Dense layer with a sigmoid activation for binary classification.
* Training: Training the model using fit_generator, with data fed through generators that yield batches of preprocessed images.
* Data Augmentation: Implementing data augmentation techniques using Keras' ImageDataGenerator to increase the diversity of the training set through random transformations like rotation, translation, and flipping.
* Regularization: Adding Dropout layers to the network to further reduce overfitting.
* Evaluation: Plotting training and validation accuracy and loss curves to assess model performance, and comparing results with and without regularization techniques.

This assignment provides practical exposure to building and training CNNs, emphasizing data preparation, model architecture, and evaluation techniques in deep learning.

### Assignment 7 - Classifying CIFAR-100 Images with Convolutional Neural Networks
This assignment involved designing and implementing convolutional neural networks (CNNs) for classifying images from the CIFAR-100 dataset. The key tasks included:

* Dataset Preparation: Dividing the CIFAR-100 training dataset into a sub-training set and a validation set, allocating 20% of the training data for validation.
* Model Design: Developing CNN models to predict the "fine" labels (class) of the images. The models were designed using the Keras library, with experimentation across various architectures, activation functions, optimizers, and hyperparameters.
* Model Selection: Identifying the top three models based on their performance on the validation set.
* Full Training and Testing: Retraining the selected models on the entire training set and evaluating their performance on the test set.
* Benchmarking: Comparing the model accuracies against other models listed on the CIFAR-100 image classification leaderboard, focusing on those that did not use extra training data.
* Reporting: Documenting the architectures, hyperparameters, and performance metrics of the three best models, including a comprehensive analysis of their test accuracies and benchmarking results.

This assignment showcases the application of CNNs to a challenging image classification problem, emphasizing model experimentation, evaluation, and reporting.

### Assignment 8 - Sequence Learning with LSTM Networks: Arithmetic Operations and Data Reversal
In this assignment, the goal was to implement an Encoder-Decoder model using LSTM (Long Short-Term Memory) networks for sequence learning tasks, focusing on arithmetic operations and data reversal. Key tasks included:

* Data Generation: Creating pairs of queries and answers for basic arithmetic operations (addition and subtraction) between numbers, formatted to maintain fixed lengths.
* One-Hot Encoding: Implementing a function to convert the query and answer strings into a one-hot encoding format for training the model.
* Model Architecture: Building an Encoder-Decoder LSTM network with:
    * An encoder LSTM layer to process the input sequences.
    * A RepeatVector layer to repeat the encoded vector for the decoder input.
    * A decoder LSTM layer to generate the output sequence.
    * A Dense layer with a softmax activation to output the final predictions.
* Training and Evaluation: Training the model on a dataset split into training, validation, and test sets, using categorical crossentropy loss and accuracy metrics. Evaluating the model on the test set to assess performance.
* Data Reversal Experiment: Reversing the query and answer strings in the dataset to investigate its impact on model performance, followed by retraining and evaluation.
* Visualization: Plotting validation accuracy over epochs for both the baseline (non-reversed) and reversed datasets to compare performance trends.

This assignment highlights the use of LSTM networks for handling sequence data, the impact of data preprocessing on model performance, and the importance of experimental validation in machine learning workflows.











