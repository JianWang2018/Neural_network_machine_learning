# Neural network for machine learning
Realized the projects of coursera course "Neural Networks for Machine Learning", which was taught by Prof. Geoffrey Hinton.
The code is written by python instead of matlab, which is the default for this course.

### Programming Assignment 1: The perceptron learning algorithm
Use iteration numbers equal to 5 as examples

Reuslt for dataset1:

<img src="nn_prj1_data1.png" height=80% width=80%>

Reuslt for dataset2:

<img src="nn_prj1_data2.png" height=80% width=80%>

Reuslt for dataset3:

<img src="nn_prj1_data3.png" height=80% width=80%>

Reuslt for dataset4:

<img src="nn_prj1_data4.png" height=80% width=80%>

The left plot shows the dataset and the classification boundary given by the weights of the perceptron. The negative examples are shown as circles while the positive examples are shown as squares. If an example is colored green then it means that the example has been correctly classified by the provided weights. If it is colored red then it has been incorrectly classified. The middle plot shows the number of mistakes the perceptron algorithm has made in each iteration so far. The right plot shows the distance to some generously feasible weight vector if one has been provided (note, there can be an infinite number of these). Points that the classifier has made a mistake on are shown in red, while points that are correctly classified are shown in green.


##### We can see that data1, data2 and data3 can be linearly separable, while data4 can not.
