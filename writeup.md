# **Traffic Sign Recognition**

This document serves as a more consitent explanation of the
steps executed in the Jupyter Notebook. It is also a requirement
for submission to the Udacity program.

This is document is part of my [git repository](https://github.com/putza/CarND-Traffic-Sign-Classifier-Project) for this project.

---

**Project OBjective: Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/fig01_example_signs.png
[image2]: ./examples/fig02_train_hist_class.png
[image3]: ./examples/fig03_train_pipeline_aug.png
[image4]: ./examples/fig04_model_comparison.png
[image4a]: ./examples/fig04_model_comparison_aug.png
[image_aug_01]: ./examples/fig01a_example_augmented.png
[image_aug_02]: ./examples/fig01b_example_augmented.png
[image_aug_03]: ./examples/fig02_train_hist_class_aug.png
[image_real_pipeline]: ./examples/fig05_real_gray.png
[image_real_classification]: ./examples/fig06_real_gray_classification.png
[image_real_allprob]: ./examples/fig07_real_gray_propability.png

[image_real_pipeline_aug]: ./examples/fig05_real_gray_aug.png
[image_real_classification_aug]: ./examples/fig06_real_gray_classification_aug.png
[image_real_allprob_aug]: ./examples/fig07_real_gray_propability_aug.png

[image_featuremap_con01]: ./examples/fig08_real_gray_featuremaps_conv01.png
[image_featuremap_con02]: ./examples/fig08_real_gray_featuremaps_conv02.png

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/putza/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the scipy and pandas libraries to calculate summary statistics of the traffic signs data set. The following output is directly taken from the code.

    Number of training examples = 34799
    Number of validation examples = 4410
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

* Example image of each class

    ![alt text][image1]
* Histogram of training dataset w.r.t. classes
*
    ![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

**Processing Pipeline**

I created an imaging pipeline, where a boolean flag activates specific parts of the pipeline. This was mainly to create a RGB and a grayscale version of all the model used in this project.

The pipeline contained these steps in order:

1. Grayscale conversion

   The grayscale conversion is done in the most simple way: The mean of all color channels. OpenCV would use a weighted average to take the eye's sensitivity to different colour channels into account. However for our purposes the average works just as well.
2. Normalization

   Normalize the image to -0.5 and 0.5

The two steps of the training pipeline are demonstrated below:

![Training Pipeline][image3]

**Data Augmentation**

I have implemented a simple data augmentation algorithm. It uses the OPenCV wrap function. The wrap function is based on a random transformation of three reference points. The image below shows the application of repeated augmentation operations on the original image in the top left.
![alt text][image_aug_01]

The augmentation applied to five random samples. Top row are original images, bottom row are augmented images.
![alt text][image_aug_02]

The category histogram of the augmented data is whown below.
![alt text][image_aug_03]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I implemented the following neural network for this project:
* LeNet for RGB and grayscale (function takes the channels as input parameter)
* SermaNet (RGB and Grayscale version)

  I followed the [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) recommended by Udacity.


SermaNet outperforms LeNet with similar computational speed. I created a fucntion which creates this model and takes the number of channels as parameter. The other hyperparameters are the mean and standard deviation of the random weight initialization.

| Layer         		|     Description	        					|
|:---------------------|:---------------------------------------------:|
| Input         		    | 32x32x3 RGB image  or 32x32x1 Grayscale image |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6 	|
| Activation:RELU 		  |	Simple RELU activation											|
| Max pooling	          | 2x2 stride,  outputs 14x14x6 		        		|
|
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16 |
| Activation:RELU 		  |	Simple RELU activation										|
| Max pooling	          | 2x2 stride,  outputs 5x5x16 		        		|
|
| -- Convolution 5x5    | 1x1 stride, valid padding, outputs 1x1x400 |
| -- Activation:RELU 	  |	Simple RELU activation
| -- Flatten            | Input = 1x1x400. Output = 400              |
|
| ++ Flatten            | Input = 5x5x16. Output = 400               |
|
| Concat [--,++]        | Input = 400 + 400. Output = 800            |
|
| Dropout               | 50% |
|
| Fully Connected       | Input = 800. **Output = 43** |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used the dataset provided by Udacity. This dataset is already split into 3 parts:

* Training data
* Validation data
* Testing data

I assumed that the va;idation data was chosen on purpose, so I did not use the scikit-learn functions to split the datasets further. Specifically if using an augmented dataset, I augmented only the training dataset.

**Optimizer**

I used the *Adam* optimizer with a learning rate of 0.001. This seemed to work out of the box.

**Algorithm Parameters**

* Epochs: 500, no specific reason. I wanted a nice plot.
* Batchsize: 1024, seemed to run fine on my graphics card

**Hyperparameters**

I provided an interface in the model, but did not use it. Used the same values as in the UDacity lectures.

* mean: 0.0
* std: 0.1
* dropout propability: 0.5


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


If an iterative approach was chosen:
* I first tried LeNet for grascale. THis worked well on digits, so it should do well on traffic signs. It did well, but did not reach the required accuracy.
    * Decision: different architecture or data augmentation

      I chose a going to a different architecture based on the provided reference. I was sure at this point that data augmentation will work as some of the traffic sigs are extremely underrepresented in the traing set. However I was curious, if a new network can get me over the desired accuracy value of 0.93.
* Next I tried to add the colour information into the Lenet network.

  This gave me perhaps a little more, but still not enough to get me over the threshold. On some runs it actually behaves worse.
* Next I used the architecture recommended (nicknamed SermaNet). The grayscale version got me over the threshold after about 100 epochs.
* Out of curisity I also tried the color version. Behaves about the same.

**No Data Augmentation**

![alt text][image4]

    LeNet-RGB:	 Final Validation Accuracy 0.9145124731690976, Testing Accuracy 0.9121140138270453
    LeNet-Gray:	 Final Validation Accuracy 0.9027210894355427, Testing Accuracy 0.9034045928258888
    SermaNet-RGB:	 Final Validation Accuracy 0.9616780039945156, Testing Accuracy 0.9461599360452412
    SermaNet-Gray:	 Final Validation Accuracy 0.957142858710689, Testing Accuracy 0.9415676965566259

**With Data Augmentation**

![alt text][image4a]

    LeNet-RGB:	 Final Validation Accuracy 0.9210884374015186, Testing Accuracy 0.9148851945677733
    LeNet-Gray:	 Final Validation Accuracy 0.9294784569145601, Testing Accuracy 0.9062549491204257
    SermaNet-RGB:	 Final Validation Accuracy 0.9761904766770447, Testing Accuracy 0.9589073640150582
    SermaNet-Gray:	 Final Validation Accuracy 0.9666666655854033, Testing Accuracy 0.9474267609612109

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:
The first column is the original image, the second the projection onto a 32x32 grid and the third column shows the imaging pipeline applied.

![Real Images][image_real_pipeline]

Potential difficulties in the images:

* 1,2: Dominating green background
* 3,4,5: Dominating blue background. Originally I had some issues with image 3 so I created a cropped version.
* 6: Confusing background


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


**Original Dataset**



![Original Prediction][image_real_classification]

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares a  little lower to the accuracy on the test set of 93%

The stop sign was identified incorrectly, with the algorithm predicting the wrong class with 99% accuracy.

**Augmented Dataset**

![Original Prediction][image_real_classification_aug]

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares a  little lower to the accuracy on the test set of 97%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the orogonal as well as the augmented data, the algorithm is almost 100% certain in all cases, even the misclassified ones. Only the logscale shows some differences.

**Original Dataset**

The top five predictions of each image are shown in the previous section.
The propability for each class if shown below on a logarythmic scale.

![Original Prediction][image_real_allprob]

The only dataset where other signs accur with a propabiliy of higher than
10E-12 is the misclassified sign in row three.

**Augmented Dataset**

The top five predictions of each image are shown in the previous section.
The propability for each class if shown below on a logarythmic scale.

![Original Prediction][image_real_allprob]

The only dataset where other signs accur with a propabiliy of higher than
10E-12 is the misclassified sign in row three.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The image below shows the output of the first convolution layer.
![Convilution 1][image_featuremap_con01]
The detected features seem to be:
* Feature Map 0,2,3: White background color within the sign
* Feature Map 1,4: The white rim around the sign
* Feature Map 5: The 70 letters in the sign

The image below shouws the 16 feature maps of layer two.
![Convolution 2][image_featuremap_con02]
* Not sure what  we see here.