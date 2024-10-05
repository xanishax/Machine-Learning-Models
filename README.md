# Machine-Learning-Models
 
1.	What is Word Embedding? Suppose you have a social media post that says, "I like Data Science because Data Science is very cool," and you use this post to train a Word2Vec model. Walk through each step of Word2Vec with detailed calculations and explanations using the following assumptions:
a. The context window size is 1.
b. The dimension of the output vectors is 2.
c. Use online learning (i.e., update the parameters after each record).
d. Each time you update the parameters, use a learning rate of 0.1 (you decide whether to add +0.1 or subtract -0.1).
 
2.	You are given the following training dataset of 4 students, recording the number of hours they spent per week, their number of course website visits per week, and the outcome of the course (either pass (1) or fail (0)): 
Time spent per week (h) 	Course website visits per week  	Pass/Fail (1/0) the course 
10 	15 	1 
15 	12 	1 
6 	5 	0 
8 	8 	0 
a.	Use the Perceptron neural network to train a model for predicting the outcome of the course. Provide detailed steps on how you train the model using each data point. Notes: Normalise the data and specify your assumptions about the learning rate, activation functions, number of iterations, and initial parameters. 
b.	Based on your trained model, would a student who spends 12 hours per week studying and visits the course website 10 times pass the course? Justify your answer.

3.	The average house prices in Mawson Lakes, Australia, over the last 3 years in million dollars are 1.1, 1.2, and 1.0, respectively. 
a.	Use an RNN to predict the house price in Mawson Lakes in 2025. Explain the details of the training process and the prediction. 
b.	Replace the RNN with an LSTM and repeat the prediction. 
Please specify your own assumptions regarding the number of iterations, learning rate, parameters, and activation functions. 
 
4.	The dataset used in this project is the Toxic Comment Classification Challenge from Kaggle. It contains approximately 159,000 comments from Wikipedia talk pages that have been labeled by human annotators as toxic or non-toxic. The dataset includes six different types of toxicity: toxic, severe toxic, obscene, threat, insult, and identity hate. 
a.	Binarize the data into Toxic (1) and Non-toxic (0). Clean the text by removing punctuation, converting it to lowercase, and removing stop words. 
b.	Train a Word2Vec model using the entire dataset with the following parameters: Vector Size = 100, Window Size = 5, Minimum word count = 5, Workers = 4. Update the word embeddings using an FNN with 2 hidden layers of 64 neurons each and a ReLU activation function.
c.	Split the data into 80% for training and 20% for testing, maintaining the same distribution of the class labels. 
d.	Train the following models: Logistic Regression (LR), FNN, RNN, LSTM, and Attention Mechanism (AM) using these parameters: 2 hidden layers with 64 neurons each, ReLU activation for hidden layers, 1 output neuron with sigmoid activation, Adam optimizer, learning rate of 0.001, 
5 epochs, 32 batch size, and metrics including accuracy, precision, recall, and F1-score. 
e.	Compare the performance of the models on the test set in terms of accuracy, precision, recall, and F1 score. 

The solution can be found here:
https://colab.research.google.com/drive/1jbbGo_9ITS4vH4ysSH1bcxDsaTI9pLph?usp=sharing
