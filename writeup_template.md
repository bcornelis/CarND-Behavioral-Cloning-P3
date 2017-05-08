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

**Status of the Project

It took me hours, weeks, months to finish the project as I was missing the problematic point of my solution.
The solution I had was:
* I started with the exact NVidia model (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) only including some dropout layers to it. 
* tried with various augmentation techniques
* tried with different parameters (learning rate, kernel sizes, ...)
* tried with different data sets (including/excluding normal driving behaviour, redirecting the the middle of the road, ...)
* tried with different training techniques (generators, full data, ...)

it simply did not work.

Then I hit on the forum mentionings of 'models being too complex'. As I was expecting the NVidia model to be 'the' model working perfectly well, I didn't initially realized it might the the model causing all the problems. Changing the model to an easier version immediatelly showed greatly improved loss and training behaviour.

[//]: # (Image References)

[final_model]: ./report_images/final_model_keras.png
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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
* I've used 'elu' activations instead of 'relu' as they seem to perform a little better according to http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/
* I've used average pooling as manual tests showed they performed better thay the max pooling algorithm
* I've used a pooling size of 7x7
* one convolutional layer is included with a depth of 64 and size of 7x7
* There's one fully connected layer of 20 nodes
* an output layer of 1 node representing the steering angle
* Dropout rate of 0.6 is included in the dense layers

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

For the training data I initially used 3 driving laps in the correct direction, 3 driving laps in the opposite direction and about 1/2 track for recovery driving.
As I had problems getting an initial version to work, I went back to the data provided by Udacity. As this data was enough to succeed the project, I didn't use any other data anymore.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

When I realized the full NVidia model wouldn't solve the problem, I started simplying the model randomly by removing some convolutional layers, some dense layers, decrease the size of the dense layers, add dropout, change the depth of the convolutional layers, etc. Quite quickly I ended up with a model which did the trick of driving the car around the track.

But then the question is: can't we do better? Can't we find a simpler model (easier/faster to train) to perform the job.

So I've written some helper methods as you can see in cell X in the jupyter notebook:
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

this method will iterate over all possible combinations of those parameters, for convolutional layers 1 to 4 and fully connected layers 1 to 3, using the create_parameterized_model_final method to create the real model.

As it's perfectly working, it does not seem to be feasible to find the optimal model, as it would take quite a long time.

So I changed my strategy: wouldn't it be possible to find a model with a single convolutional layer and a single dense layer to solve the problem.

Therefore, I've writting the following function:
##### find_optimal_model_oneConv_oneDense
this will iterate through all possible cobminations of a single convolutional layer and a single dense layer to find the optimal solution.

Now that I can generate 'random' models, the question is how to decide which model performs better than others. The solution is to create a generator to generate models according to the parameters requested, train this model, and see which trains best according to the loss parameters.

For this 'first level training' I used the following parameters:
* batch_size=50
* samples_per_epoch=200
* validation_samples=100
* training_set_length=0.85
* epochs: 150 (but an early stopping callback with a patience of 4)

And important callback functions:
* early stopping to make sure not to coninue fitting when already overfitting
* csv logger to log output for all epochs to files to allow for later processing

Querying those files showed the chosen model as being the best (which offcourse, as seen from the low training parameters, can be questioned). But when I used the model trained to find the 'optimal' one, it already showed some interesting drigving bevaviour, as you can see [here](./report_images/trained_validation.mp4)

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
