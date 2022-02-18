# zhelper-cvtool
a visual filter graph help you debug your cv program.
![img](https://img2020.cnblogs.com/blog/665551/202011/665551-20201125040001797-1306568990.gif)
![img](https://img2020.cnblogs.com/blog/665551/202011/665551-20201117044056777-1920859742.gif)
![img](https://img2020.cnblogs.com/blog/665551/202011/665551-20201117044204452-738904851.gif)
# make
```
g++ `pkg-config --cflags opencv3` -O3 -std=c++11 -o cvtool cvtool.cpp `pkg-config --libs opencv3`
```
```
g++ `pkg-config --cflags opencv4` -O3 -std=c++11 -o cvtool cvtool.cpp `pkg-config --libs opencv4`
```
# !!!
you can only train cascade with opencv3.4 and use them with opencv4.

aknowledge from the book `Qt 5 and OpenCV 4 Computer Vision Projects` page `231`.

you can download a windows prebuilt opencv3.4 from [here](https://github.com/huihut/OpenCV-MinGW-Build), size of 22M only.

if you do not know which prebuilt is x64 or x86, read [OpenCV-MinGW-Build/issues/19#](https://github.com/huihut/OpenCV-MinGW-Build/issues/19#issuecomment-1044967562).
# usage
```
cvtool <image> <filter,...>
```
```
cvtool Mario/%04d.png pyrDown,pyrUp,morphology,channel,canny,contours
```
```
cvtool $SomeBuildingImage morphology,bgr2gray,canny,houghlinesP
```
```
cvtool $SomeImage morphology,medianblur,bgr2gray,houghcircles
```
```
cvtool $SomeImage morphology,blob
```
```
cvtool $SomeImage cut2,cascade
```
# key
* `Q` or `q` to quit the program.
* `space` to next frame, is the input image is group of pictures, or gif, or video.
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
* morphology
  * trackbar1, ``, 0-20 
  * trackbar2, `method:`, 0-6
    * 0, `open`
    * 1, `close`
    * 2, `erode`
    * 3, `dilate`
    * 4, `gradient`
    * 5, `tophat`
    * 6, `blackhat`
  * trackbar3, `rect/ellipse/cross`, 0-2
  * trackbar4, `kernel(OFF/ON)`, 0-1
**image process**
### filter-sobel,scharr
### filter-Laplacian
### filter-canny
for houghlinesP case, take a bigger threshval-1, and a aperture size of 3, and switch on L2gradient to avoid more noise.

* canny
  * trackbar1, `threshval-1`, 0-255 
  * trackbar2, `threshval-2`, 0-255 
  * trackbar3, `aperture size |1`, 0-7 
  * trackbar4, `L2 gradient (OFF/ON)`, 0-1

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
### filter-cut2,anno
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
* contours
  * trackbar1, `threshold*.001`, 0-200 
  * trackbar2, `show poly(OFF/ON)`, 0-1
  * trackbar3, `show contours(OFF/ON)`, 0-1
  * trackbar4, `RETR:`, 0-4
    * 0, `RETR_EXTERNAL`
    * 1, `RETR_LIST`
    * 2, `RETR_CCOMP`
    * 3, `RETR_TREE`
    * 4, `RETR_FLOODFILL`

### filter-match
**apply to match template**
* match
  * trackbar1, `method:`, 0-4
    * 0, SQDIFF
    * 1, SQDIFF NORMED
    * 2, TM CCORR
    * 3, TM CCORR NORMED
    * 4, TM COEFF NORMED
  * trackbar2, `threshold*.001%`, 0-10000

### filter-cascade
**apply to cascade**
* cascade
  * trackbar1, `switch(OFF/ON)`, 0-1

### filter-feature
**apply to detect features** 
* feature
  * trackbar1, `feat count`, 0-INT_MAX
  * trackbar2, `feature:`, 0-9
    * 0, `sift`
    * 1, `orb`
    * 2, `brisk`
    * 3, `kaze`
    * 4, `akaze`
    * 5, `mser`
    * 6, `fast`
    * 7, `agast`
    * 8, `gftt`
    * 9, `blob`
  * trackbar3, `Affine (OFF/ON)`, 0-1 

