# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I first display 20 random images with thier Class IDs from the training set and followed by a hisogram of the training set with Class ID on the x axis and number/frequency of the traffic sign belonging to that class ID bucket denoted by the bar chart.

![Random Images from the training set](./writeup_images/Random_Signs.jpg)
![Histogram of training set](./writeup_images/Histogram.jpg)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I wrote my own routine image_augmentation.py in order to define functions for augmenting data. 

As is evident from the histogram (above), the classes are not balanced and one of the objectives here is to balance the classes and generate more of fake data using the existing data.

The routine defines functions like rotate_randomly, add_noise_randomly, translate, rescale_image, and flip image left-to-right and upside-down. 

I use rotate_randomly, translate and crop functions to augment the images below. A sample of image augmentation is shown in the code cell below which shows the originial image and the augmented images for the traffic-sign: Speed Limit (50km/h).

![50 kmph](./writeup_images/50kmph_Original.jpg)

![](./writeup_images/Augmented_Images.png)

I automated the process for augmenting images in the codecell below.

I choose an upper limit (900) as the frequency for each class. More fake images would be generated for any class which has samples less than 900. This trick is to ensure the classes are balanced. The upperlimit was chosen after an iterative process to see what number gives the best accuracy for the validation data.





As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image22]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|


* Layer_1: 5x5 Convolution. Input --> 32x32x1, Output --> 28x28x6
* Activation_1: ReLu(Layer_1)
* Max Pooling Layer_1: Input --> 28x28x6, Output --> 14x14x6 with 2x2 filter and stride of 2
* Layer_2: 5x5 Convolution. Input --> 14x14x6, Output --> 10x10x16
* Activation_2: ReLu(Layer_2)
* Max Pooling Layer_2: Input --> 10x10x16, Output --> 5x5x16 with 2x2 filter and stride of 2
* Layer_3: 5x5 Convolution. Input --> 5x5x16, Output 1x1x400
* Activation_3: ReLu(Layer_3)
* Flatten Layer_2 and Layer_3 both with size 400
* Concatenate flattened Layer_2 and Layer_3, size 800
* Dropout
* Fully Connected Layer_4: Input --> 800, Output --> 43
* Logits = Output of Fully Connected Layer_4


 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* Layer_1: 5x5 Convolution. Input --> 32x32x1, Output --> 28x28x6
* Activation_1: ReLu(Layer_1)
* Max Pooling Layer_1: Input --> 28x28x6, Output --> 14x14x6 with 2x2 filter and stride of 2
* Layer_2: 5x5 Convolution. Input --> 14x14x6, Output --> 10x10x16
* Activation_2: ReLu(Layer_2)
* Max Pooling Layer_2: Input --> 10x10x16, Output --> 5x5x16 with 2x2 filter and stride of 2
* Flatten Layer: Flatten the output shape of the final pooling layer such that it's 1D. Output --> 400
* Fully Connected Layer_1: Input --> 400, Output --> 120
* Activation_3: ReLu(Fully Connected Layer_1)
* Dropout
* Fully Connected Layer_2: Input --> 120, Output --> 84
* Activation_4: ReLu(Fully Connected Layer_2)
* Dropout
* Fully Connected Layer_3: Input --> 84, Output -->83
* Logits = ouput of Fully Connected Layer_3



* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?

The model uses AdamOptimizer for both LeNet and Stage_2_LeNet. Following values were used for the hyperparameters:

* BATCH_SIZE: 100
* EPOCHS: 60 
* Learning rate: 0.0010
* mu(mean) = 0.0
* sigma(SD) = 0.1
* dropout(keep_prob) = 0.5

As such these parameters were fixed after trying out a lot of different while tuning the hyperparameters.



* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Left      		| Keep Left   									| 
| Speed Limit(20kmph)	| Speed Limit(20kmph)							|
| Yield					| Yield											|
| Speed Limit(30kmph)	| Speed Limit(30kmph)			 				|
| Children crossing		| Children crossing    							|
| Turn left ahead		| Turn left ahead    							|
| Roundabout mandatory	| Roundabout mandatory 							|
| Speed Limit(100kmph)  | Speed Limit(100kmph) 							|


The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This is far better when compared to the accuracy on the test set of 93.1%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell with heading "Predict the Sign Type for Each Image" in the Ipython notebook.

The model is absolutely sure about all the 8 images. The top five softmax probabilities are shown below for each image in the new dataset.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0,0,0,0,0  			| Keep Left   									| 
| 1.0,0,0,0,0			| Speed Limit (20kmph)							|
| 1.0,0,0,0,0			| Yield											|
| 1.0,0,0,0,0  			| Speed Limit (30kmph)			 				|
| 1.0,0,0,0,0		    | Children crossing   							|
| 1.0,0,0,0,0		    | Turn left ahead      							|
| 1.0,0,0,0,0		    | Roundabout mandatory 							|
| 1.0,0,0,0,0		    | Speed Limit (100kmph)							|