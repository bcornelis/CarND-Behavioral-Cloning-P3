# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

**Status of the Project**

It took me hours, weeks, months to finish the project as I was missing the problematic point of my solution.
The solution I had was:
* I started with the exact NVidia model (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) only including some dropout layers to it. 
* tried with various augmentation techniques
* tried with different parameters (learning rate, kernel sizes, ...)
* tried with different data sets (including/excluding normal driving behaviour, redirecting the the middle of the road, ...)
* tried with different training techniques (generators, full data, ...)

it simply did not work.

Then I hit on the forum mentionings of 'models being too complex'. As I was expecting the NVidia model to be 'the' model working perfectly well, I didn't initially realized it might the the model causing all the problems. Changing the model to an easier version immediatelly showed greatly improved loss and training behaviour.

Unfortunately as so many time is lost, and there's still some important lessons and the final project left, I don't have the time to finish this project as I would. It's working, but there are some shortcommings.

[//]: # (Image References)

[final_model]: ./report_images/final_model_keras.png
[augmentation_left-right]: ./report_images/augmentation_left-right.png
[augmentation_flipping]: ./report_images/augmentation_flipping.png
[augmentation_brightness]: ./report_images/augmentation_brightness.png
[augmentation_translation]: ./report_images/augmentation_translation.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the following layers:
![alt text][final_model]

Important nodes about this model:
* The size of the images is scaled to (100,33) as I didn't notice any visual decrease in driving performance
* I've used 'elu' activations instead of 'relu' as they seem to perform a little better according to http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/
* I've used average pooling as manual tests showed they performed better thay the max pooling algorithm
* I've used a pooling size of 7x7
* one convolutional layer is included with a depth of 64 and size of 7x7
* There's one fully connected layer of 20 nodes
* an output layer of 1 node representing the steering angle
* Dropout rate of 0.6 is included in the dense layers

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

For the training data I initially used 3 driving laps in the correct direction, 3 driving laps in the opposite direction and about 1/2 track for recovery driving.
As I had problems getting an initial version to work, I went back to the data provided by Udacity. As this data was enough to succeed the project, I didn't use any other data anymore.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

When I realized the full NVidia model wouldn't solve the problem, I started simplying the model randomly by removing some convolutional layers, some dense layers, decrease the size of the dense layers, add dropout, change the depth of the convolutional layers, etc. Quite quickly I ended up with a model which did the trick of driving the car around the track.

But then the question is: can't we do better? Can't we find a simpler model (easier/faster to train) to perform the job.

So I've written some helper methods as you can see in cell 05 in the jupyter notebook:
##### create_parameterized_model_final
will create the following model:
* a normalization layer 
* a cropping layer removing the top and bottom parts of the image 
* convolutional layers as specified by a paramter. parameters to tweak per convolutional layer are the depth, size, activation function, and pooling size 
* fully connected layers as specified by a parameter. parameters to tweak per fully connected layer are the number of nodes, activation function and dropout rate.
* a finaly layer representing the output layer with a single node

##### find_optimal_model
As the NVidia model was chosen as a reference model, I still wanted to fit the idea: some convolutional layers followed by some fully connected layers. This method will generate all possible combinations for specific parameters I would like to tune. Those parameters are:
* conv_kernel_sizes = [8,16,32,64]
* conv_filter_sizes = [2,3,5,7]
* activation_functions = ['elu']
* pooling_functions = ['avg']
* pooling_sizes = [2,3,5,7]
* dense_units = [1000,100,50,20, 0]
* dropout_rates = [0.6]

this method will iterate over all possible combinations of those parameters, for convolutional layers 1 to 4 and fully connected layers 1 to 4, using the create_parameterized_model_final method to create the real model.

Even it's perfectly working, it does not seem to be feasible to find the optimal model, as it would take quite a long time.

So I changed my strategy: wouldn't it be possible to find a model with a single convolutional layer and a single dense layer to solve the problem.

Therefore, I've writting the following function:
##### find_optimal_model_oneConv_oneDense
this will iterate through all possible cobminations of a single convolutional layer and a single dense layer to find the optimal solution.

Now that I can generate 'random' models, the question is how to decide which model performs better than others. The solution is to create a generator to generate models according to the parameters requested, train this model with a very small batch size, and see which trains best according to the loss parameters.

For this 'first level training' I used the following parameters:
* batch_size=50
* samples_per_epoch=200
* validation_samples=100
* training_set_length=0.85
* epochs: 150 (but an early stopping callback with a patience of 4)

And important callback functions:
* early stopping to make sure not to coninue fitting when already overfitting
* csv logger to log output for all epochs to files to allow for later processing

Querying those files showed the chosen model as being the best (which offcourse, as seen from the low training parameters, can be questioned). But when I used the model trained to find the 'optimal' one, it already showed some interesting drigving bevaviour, as you can see [here](./report_images/trained_validation.mp4). The model is included [here](./report_images/trained_validation.hdf5)

Training this model with more samples (and only left/right, and flipping as augmentation techniques) I ended up by driving the car around the track. The result can be seen [here](./report_images/final.mp4) and the model is included [here](./report_images/final.hdf5)

As information: I know that this method of finding the 'optimal' model is definitely not computationally correct in any way, but it allowed me not only to manually tweak some random properties of the NVidia model, but play around with different models etc.

#### 3. Creation of the Training Set & Training Process
Augmentation techniques can be found in cell 02.

##### Augmentation
I've implemented the followign augmentation algorithms.
![][augmentation_left-right]

###### Left/Right camera images
Also use the left and right camera images. A correction factor of 0.25 is applied to correct the steering angle for the side images.
![][augmentation_left-right]

###### Image Flipping
I there's an image with a steering angle to the left, we can flip the angle and we have an extra training sample with an image steering to the right.
![][augmentation_flipping]

###### Brightness Augmentation
The result of applying this transformation is the same image with a different brightness factor applied.
![][augmentation_brightness]

All credits of this implementation go to: Vivek Yadav (https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)

###### Translation Augmentation
The result is that the image is shifted left/right or top/bottom.
![][augmentation_translation]

All credits of this implementation go to: Vivek Yadav (https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)

##### Generator
I used a generator to provide the training data (cell 06). The logic of the generator is:
1. get a random line in the training csv data
2. get a random image from this line (center, left or right)
3. only process straight images in 10% of the cases
4. scale the image to fit the expected input size of the model. I scale to (33+95,100) for which the 95 is the part being cropped off in the mode itself
5. adjust the steering if left or right image
6. flip the image in 50% of the cases, and if it's not a straight one
7. apply brightness augmentation in 90% of the cases
8. apply translation/shifting augmentation in 80% of the cases

##### Validator
For validation, a generator is used generating tuples according to the following steps (cell 06):
1. get a random line in the training csv data
2. get the center image from this line
3. scale the image to fit the expected input_size of the model

###### Augmentation
While playing around, I've implemented the following augmentation algorithms:

##### Training and validation set
I've used 2 training configurations:
one to find the 'optimal' model, and one to train this model. In both situations I used the same training data generator and validation data generator, but the following parameters were different:
finding the best model:
* batch_size=50
* samples_per_epoch=200
* validation_samples=100
* training_set_length=0.85

and for training this model:
* batch_size=400
* samples_per_epoch=40000
* validation_samples=8000
* training_set_length=0.85

All csv lines were shuffled before training, split into training and validation set and the generators randomly pick elements form the apropriate list.

for the optimal model I did not use brightness augmentation and translation augmentation. I compared the final model generation with using those techniques and without, but there was no big visual improvement.

## Final Movie
The final movie can be seen [here](./report_images/final.mp4)
