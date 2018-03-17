## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test\_video.mp4 and later implement on full project\_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./ouput_images/clor%20channel%20images.png
[image2]: ./ouput_images/clor%20channel%20images1.png
[image3]: ./ouput_images/clor%20channel%20images2.png
[image4]: ./ouput_images/clor%20channel%20images4.png
[image5]: ./ouput_images/color%20histogram.png
[image6]: ./ouput_images/color%20histogram1.png
[image7]: ./ouput_images/color%20histogram2.png
[image8]: ./ouput_images/color%20histogram3.png
[image9]: ./ouput_images/heatmap.png
[image10]: ./ouput_images/heatmap1.png
[image11]: ./ouput_images/heatmap2.png
[image12]: ./ouput_images/heatmap3.png
[image13]: ./ouput_images/heatmap4.png
[image14]: ./ouput_images/heatmap5.png
[image15]: ./ouput_images/hog%20image.png
[image16]: ./ouput_images/hog%20image1.png
[image17]: ./ouput_images/hog%20image2.png
[image18]: ./ouput_images/hog%20image3.png
[image19]: ./ouput_images/test%20res.png
[image20]: ./ouput_images/test%20res1.png
[image21]: ./ouput_images/test%20res2.png
[image22]: ./ouput_images/test%20res3.png
[image23]: ./ouput_images/test%20res4.png
[image24]: ./ouput_images/test%20res5.png
[image25]: ./ouput_images/training%20data%20visualization.png
[image26]: ./ouput_images/training%20data%20visualization1.png
[image27]: ./ouput_images/training%20data%20visualization2.png
[image28]: ./ouput_images/training%20data%20visualization3.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points ### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

\-\-\-
### MODEL TRAINING
#### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the ```get_hog_features``` function definition in the IPython notebook 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some examples of images in each of the `vehicle` and `non-vehicle` classes:

![alt text][image25]

![alt text][image26]

![alt text][image27]

![alt text][image28]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
After trying various parameters I arrived on the following set of parameters for the HOG transformations:

| HYPERPARAMETRS|VALUE|
|:--------------:|:------------:|
|Orientations|12|
|Cells Per Block|2|
|Pixel per cell|10|
|Hog Channel| ALL|

Below are few random images obtained by using the above mentioned parameters:

![alt text][image15]

![alt text][image16]

![alt text][image17]

![alt text][image18]

#### 2. Explain how you settled on your final choice of HOG parameters.

First I tried to obtain a result with the initial hyperparameters that are mentioned in the classroom for YUV color space. But the result was not satisfactory for these set of parameters. I then switched to the YCrCb color space for better results. Then I tried increasing the orientations for each cell, that improved the result but the number for features increased a lot. Then to reduce the number of features and to find a balance between the good set of features without increasing the feature length a lot i arrived at the parameters mentioned in the table above.

#### Color Histogram:
I have also used to the color histogram to train the model. I have used  the YCrCb color space and a 32 histogram bins

| HYPERPARAMETRS|VALUE|
|:--------------:|:------------:|
|Color Space|YCrCb|
|Hist Bins|32|

Below is the image of the color histogram:

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the above parameters and the 80% of the data set and keeping the 20% of the data set for testing the trained SVM. I obtained a total feature vector length of 3696 parameters after combing all the HOG and the color histogram features. With this feature vector length I obtained a accuracy of 96.96% on the test set. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have implemented the HOG subsampling window-search for finding the car positions using the trained classifier. for the subsampling I have taken various sub strips of the image and a scale factor corresponding to each of the sub strips. I have taken the sub strips so as to minimize the scans for the cars. I have taken smaller strips in the far end of the road with a smaller scale factor of the windows as the cars at the far end of the road will be smaller than the other cars. I have kept this thing in mind making the strips and the scale factor for the windows, I have also used a starting point on the x axis as we are not concerned about the cars on the other side of the road. This increased the performance time of the classifier while searching for the images

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 5 scales using YCrCb color space and 3-channel HOG features plus histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image19]

![alt text][image20]

![alt text][image21]

![alt text][image22]

![alt text][image23]

![alt text][image24]

\-\-\-

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video/project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used \`scipy.ndimage.measurements.label()\` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. For the video I have taken the sum of the previous 10 frames so that the detection if the car is more accurate. 

### Here are the heatmaps for the six test images:

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

\-\-\-

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced a hard time figuring out the right set of parameters that would work the best for me. But still there are some place in the video where the pipeline fails. In the next few lines I will discuss where all the pipeline fails on the video. 
As seen the video the pipeline fails in some of the frames gives a false detection or does not detect the car when it is there in the frame.
This could be due to the scales that I am using for the subsampling process or due the threshold of the heat map that nullifies the detection of the classifier as it not that strong as it should be. 
To make the pipeline more robust I could train the classifier better so that the detection at the places where car is there in the frame improves. To train the classifier I could also use the spatial features as it might help to train the classifier better and that in turn will result in better accuracy on the test images and the video.  
I could also try using a non linear kernel for better results.
