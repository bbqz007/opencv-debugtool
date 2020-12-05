# zhelper-cvtool
a visual filter graph help you debug your cv program.
![img](https://img2020.cnblogs.com/blog/665551/202011/665551-20201125040001797-1306568990.gif)
![img](https://img2020.cnblogs.com/blog/665551/202011/665551-20201117044056777-1920859742.gif)
![img](https://img2020.cnblogs.com/blog/665551/202011/665551-20201117044204452-738904851.gif)
# make
```
g++ `pkg-config --libs --cflags opencv3` -std=c++11 -o cvtool cvtool.cpp` 
```
# usage
```
cvtool <image> <filter,...>
```
```
cvtool Mario/%04d.png pyrDown,pyrUp,morphology,channel,canny,contours
```
# custom program
```
#include "cvtool.h"
using namespace zhelper;
using namespace cvtool;
...
filter_graph fg;
fg.apply([](Mat res) { /** detect object or other things */}).filter(frame);
```
1. implement your filter derived from itf_filter;
2. or use filter_graph to apply your function which accepts Mat filtered by filter_graph;
# filters
```
    BRANCH(Canny);
    BRANCH(threshold);
    BRANCH(morphology);
    BRANCH(Canny);
    BRANCH(medianBlur);
    BRANCH(GaussianBlur);
    BRANCH(blur);
    BRANCH(bilateral);
    BRANCH(box);
    BRANCH(sqrBox);
    BRANCH(Sobel);
    BRANCH(Scharr);
    BRANCH(Laplacian);
    BRANCH(pyrDown);
    BRANCH(pyrUp);

    BRANCH(warpAffine);
    BRANCH(warpPerspective);
    BRANCH(warpPolar);

    BRANCH(cornerMinEigenVal);
    BRANCH(cornerHarris);
    BRANCH(cornerEigenValsAndVecs);
    BRANCH(preCornerDetect);
    BRANCH(norm_minmax);

    BRANCH(HoughLines);
    BRANCH(HoughLinesP);
    BRANCH(HoughCircles);

    BRANCH(channel);
    BRANCH(bgr2gray);
    BRANCH(gray2mask);
    BRANCH(range);
    BRANCH(colormap);
    BRANCH(cut);
    BRANCH(cut2);
    BRANCH(anno);
    BRANCH(crop);
    BRANCH(zoom);

    BRANCH(deskew);
    BRANCH(dem);
    BRANCH(distrans);

    BRANCH(contours);
    BRANCH(match);
    BRANCH(cascade);
    
    BRANCH(feature);
    BRANCH(blob);
```
### filter-morphology,medianBlur,GaussianBlur,blur,bilateral,box
**image process**
### filter-sobel,scharr
### filter-Laplacian
### filter-canny
### filter-cornerMinEigenVal,cornerHarris,cornerEigenValsAndVecs,preCornerDetect
**corner detector**

output CV_32F Mat which is computed description.

### filter-normalize
**normalize CV_32F Mat**

when you use cornerXXX filter, you can get a output CV_32F Mat of result.

you could use this filter to normalize the values of the CV_32F Mat, minVal to 0, maxVal to 1, others to 0.xxx. 

### filter-channel,bgr2gray
**output gray or CV_8U mat**

### filter-gray2mask
a mask just has 0s or 255s, using your given threshold.

### filter-range
**hsv,hls,bgr channels in ranges**

### filter-cut
**clip and save, or apply to your custom handler**

select a region to clip, **double click right_mouse_button** to save picture, default as png, with ctl as jpg.
### filter-crop
**clip but not save, and pass it to next filter**

select a region to clip, **double click right_mouse_button** to pass it to next filter.
### filter-cut2
**similar to opencv_annotation.**

help you easily build samples for machine learning. 

**ctrl+left_mouse_button** to select positive region, can select multiple, and **double click right_mouse_button** to generate images to pos/ and neg/, and infos to pos/pos.txt and neg/neg.txt. helpful for using opencv_createsamples. 

**double click left_mouse_button** can reuse last region size and anchor to the point you are clicking.

**middle_mouse_button** clears what you select before generating stuffs.

when you `cvtool image cut2,cascade` to check your trained classifier, you can easily use cut2 filter to add samples and train again.

### filter-zoom
**strentch x(width) or y(height)**

range in 0.5x and 4x.

use case `cvtool crop,zoom` can clip a part of picture and strentch it to make more features for matches or detections.

### filter-contours
**apply to find contours**

### filter-match
**apply to match template**

### filter-cascade
**apply to cascade**

### filter-feature
**apply to detect 2d features** 

