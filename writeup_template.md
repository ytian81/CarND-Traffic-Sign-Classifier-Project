# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./dist.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./traffic_signs/go_straight_or_right.jpg  "Traffic Sign 1"
[image5]: ./traffic_signs/no_entry.jpg "Traffic Sign 2"
[image6]: ./traffic_signs/pedestrain.jpg "Traffic Sign 3"
[image7]: ./traffic_signs/speed_limit.jpg "Traffic Sign 4"
[image8]: ./traffic_signs/turn_right_ahead.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python built-in functionality to calculate summary statistics of the traffic
signs data set:

* ```
  Number of training examples = 34799
  Number of validation examples = 4410
  Number of testing examples = 12630
  Image data shape = (32, 32, 3)
  Number of classes = 43
  ```

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

Class distribution for train data, validation data and test data
![train label distribution][image1]

As you can see, their distributions are almost the same, which is good for training a deep learning
model

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because most of time, color information is irrelevant to the traffic sign name

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I don't normalize the data because when I include image data normalization, it becomes harder to train. I cannot overfit the training set. If I don't include the normalization, I can easily get ~95% validation accuracy.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 RGB image   						|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28\*28\*6 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14\*14\*6 |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10\*10\*16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5\*5\*16 |
| Fully connected		| 400\*120        				|
| RELU				|         									|
| Dropout |												|
| Fully connected		| 120\*84        				|
| RELU				|         									|
| Dropout |												|
| Fully connected		| 84\*43        				|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an `tf.nn.softmax_cross_entropy_with_logits` loss with L-2 regularizer because I am dealing with multi-class classification problem.  In the architecture above, I also use drop out layer for regularization. I used `Adam` optimizer because it can automatically decay learning rate when learning becomes slow. (initial learning rate is 1e-3). Other hyperparamters are 20 as number of epches, 32 as batch_size and 5e-3 as regularizatio coefficient.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.8%
* validation set accuracy of 95.7%
* test set accuracy of 93.6%

If an iterative approach was chosen:
* I use LeNet used in the course as a starting point. However, I don't have any reguralization at the first iteration. So I overfit the training set, get a very high accuracy on training data, but a low accuracy on validation set. I add both L2-regularizer and drop out layer to prevent overfitting. Two hyperparamters are included, i.e. the regularization coefficient and drop out rate. I try 1e-2, 5e-3 and 1e-3 as the regularization coefficient and find 5e-3 gives me the best validation accruacy. For drop out rate, I choose 0.5 when training and 1.0 when testing.
* I choose Adam optimizer, so I don't need to worry too much about learning rate. I choose 1e-3 as the starting learning rate.
* I choose 32 as batch size. In general, the smaller batch size, the gradients become more noisy.
* I choose 20 as the nubmer of epoches so that network can be fully trained. And I save the model only when I get a better validation accuracy. Therefore, even when validation accruacy drops after certain time, I still have the best model.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 
![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8]

The first and fourth images should be easy to classify. The second, the third and the fifth are difficult
because when they are resized to 32\*32, the traffic sign can be very hard to separate from
background. The fifth image even has other traffic signs as background, though they are occulded.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Go straight or right | Go straight or right |
| No entry     	| No entry 		|
| Pedestrain	| General caution	|
| Speed limit 50 km/h	| Speed limit 80 km/h	|
| Turn right ahead	| Turn right ahead |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is not as good as prediction on test set. Two misclassified classes are `Pedestrain` and `Speed limit`. First of all, this is acceptable because we only test 5 new random images found online. The sample size is too small to argue the prediction capability of the network. If we look at the softmax probabilities, we will find the ground truth label is among the top 5 probabilities. Secondly, my wild guess of wrong classification of `Speed limit` and 'Pedestrain' are attributed to the fact that they are not follow the training data distribution. Their styles look quite different to training data.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a `Go straight or right` (probability of 0.512), and the image does contain a `Go straight or right` sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .512    | 36-Go straight or right |
| .096     	| 38-Keep right |
| .043		| 17-No entry	|
| .043	      		| 14-Stop			|
| .033				  | 28-Children crossing |


For the second image, the model is 100% sure that it is `No entry` sign. The top five soft max probabilies were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0   | 17-No entry |
| 1.2e-9 | 40-Roundabout mandatory |
| 3.3e-10	| 37-Go straight or left	|
| 3.2e-10	| 9-No passing	|
| 1.3e-10	| 0-Speed limit (20km/h) |

For the third image, the model says the probability of `General caution` is 40.5%. The probability of `Pedestrains` is still in top 5 class, which is 12.5%

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .405 | 18-General caution |
| .248   | 21-Double curve |
| .177	| 24-Road narrows on the right	|
| .125	     | 27-Pedestrians	|
| .0189			| 26-Traffic signals |

For the fourth image, the model claims it is `50 km/h speed limit` sign with 99.9% probablity. While the ground truth `80km/h speed limit` is the second most probable class.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .999       | 2-Speed limit (50km/h) |
| .0002       	| 5-Speed limit (80km/h) |
| .00002 		| 1-Speed limit (30km/h)	|
| .00001 	      		| 3-Speed limit (60km/h)	|
| .000002				  | 31-Wild animals crossing |

For the last image, the model believes it is `Turn right ahead` with 61.7% probability.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .617 | 33-Turn right ahead |
| .215   | 15-No vehicles |
| .070	| 4-Speed limit (70km/h)	|
| .041	     | 40-Roundabout mandatory	|
| .011				| 12-Priority road |
