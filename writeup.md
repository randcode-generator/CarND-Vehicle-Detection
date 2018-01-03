**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/image1.png
[image2]: ./images/image2.png
[image3]: ./images/image3.png
[image4]: ./images/image4.png
[image5]: ./images/image5.png
[scale_1_0]: ./images/scale_1_0.png
[scale_1_2]: ./images/scale_1_2.png
[scale_1_5]: ./images/scale_1_5.png
[frame1]: ./images/frame1.png
[frame2]: ./images/frame2.png
[frame3]: ./images/frame3.png
[heat1]: ./images/heat1.png
[heat2]: ./images/heat2.png
[heat3]: ./images/heat3.png


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I extracted HOG features of all 3 RGB channels from training images

| orient | pix_per_cell | cell_per_block | 
|:------:|:------------:|:--------------:| 
| 9      | 8            | 2              |

```python
hog_features = []
for channel in range(feature_image.shape[2]):
    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                        orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True))
hog_features = np.ravel(hog_features)
```

Here's an example of `vehicle` and `non-vehicle` respectively:

![image1]

![image2]

The output of the HOG of one of the channels:

![image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

The bounding boxes is the right size.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used spatial binning and HOG on all channels for training my classifier

```python
spatial = bin_spatial(image)
features.append(np.hstack((hog_features,spatial)))
```

I used default parameters for SVC training

```python
svc = LinearSVC()
svc.fit(X_train, y_train)
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

**Scale of 1:**

**Pros:** The bounding boxes are more precise

**Cons:** Slow!

![scale_1_0]

**Scale of 1.5:**

**Pros:** Faster

**Cons:** Bounding boxes are bigger and less precise

![scale_1_5]

So I chose scale of 1.2 for pretty good speed and decent bounding boxes. Like scale, I chose 2 cells per step due to speed and bounding box precision.

![scale_1_2]

```python
ystart = 400
ystop = 670
xstart = 680
xend = 1280
scale = 1.2
pix_per_cell = 8
cell_per_block = 2
orient = 9

box_list = []
draw_img = np.copy(img)
img = img.astype(np.float32)/255

img_tosearch = img[ystart:ystop,xstart:xend,:]

ctrans_tosearch = img_tosearch
if scale != 1:
    imshape = ctrans_tosearch.shape
    ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 

window = 64
nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
cells_per_step = 2  # Instead of overlap, define how many cells to step
nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

for xb in range(nxsteps):
    for yb in range(nysteps):
        ypos = yb*cells_per_step
        xpos = xb*cells_per_step

        xleft = xpos*pix_per_cell
        ytop = ypos*pix_per_cell

        subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
```

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I save images from the scaling window and use those as training images. Also, I make sure the distance is at least 0.3 from the decision line 

Here are some example images:

![image4]
![image5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./videoOutput/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

```python
# average the heatmap
global heatmap_prev
if heatmap_prev is not None:
    heatmap = (heatmap+heatmap_prev)//2.0
```

```python
# Draw bounded box only if the confidence in the
# bounded box is greater then 6.
# This filters out the boxes that has low confidence.
maxval = np.max(heatmap[y1:y2, x1:x2])
if(maxval > 6):
    cv2.rectangle(draw_img, box[0], box[1], (0,0,255), 6)

# All bounded boxes confidence increases by 2.
# This helps filter out bounded boxes that only exists in 1 frame.
heatmap[y1:y2, x1:x2] += 2
```

```python
# Filter out all the low confidence.
# Anything lower than 5 is zeroed out.
heatmap[heatmap <= 5] = 0
heatmap_prev = heatmap
```

To filter out false positives, add extra weight (by 2) to bounding boxes to increase confidence. If the confidence is above 6, draw bounding boxes. Also, zero out the low confidence (less then 6) to filter out bounding boxes that only exists on 1 frame.

### Here are three frames and their corresponding heatmaps:

| Frame     | Heat     | 
|:---------:|:--------:| 
| ![frame1] | ![heat1] |
| ![frame2] | ![heat2] |
| ![frame3] | ![heat3] |

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Challenges:
1. The bounding boxes were really unstable
2. Speed of processing images
3. Shadow caused false positives

Likely failure:
1. Different shapes of cars

More robust:
1. Try different color spaces
2. Stabilize the bounding boxes to ensure greater confidence around the vehicle

