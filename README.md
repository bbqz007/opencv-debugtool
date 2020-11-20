# zhelper-cvtool
a visual filter graph help you debug your cv program.
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
    BRANCH(medianBlur);
    BRANCH(GaussianBlur);
    BRANCH(blur);
    BRANCH(bilateral);
    BRANCH(box);
    BRANCH(sqrBox);
    BRANCH(Sobel);
    BRANCH(Scharr);
    BRANCH(Laplacian);
    BRANCH(cornerMinEigenVal);
    BRANCH(cornerHarris);
    BRANCH(cornerEigenValsAndVecs);
    BRANCH(warpAffine);
    BRANCH(warpPerspective);
    BRANCH(warpPolar);
    BRANCH(pyrDown);
    BRANCH(pyrUp);

    BRANCH(channel);
    BRANCH(bgr2gray);
    BRANCH(range);
    BRANCH(colormap);
    BRANCH(cut);
    BRANCH(crop);
    BRANCH(zoom);
    
    BRANCH(dem);
    BRANCH(distrans);
    
    BRANCH(contours)
```
