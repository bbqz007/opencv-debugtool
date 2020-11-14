# zhelper-cvtool
a visual filter graph help you debug your cv program.

# make
```
make `pkg-config --libs --cflags opencv3` -std=c++11 -o cvtool cvtool.cpp` 
```
# usage
```
cvtool <image> <filter,...>
```
```
cvtool Mario/%04d.png pyrDown,pyrUp,morphology
```
# custom program
```
filter_graph fg;
fg.apply([](Mat res) { /** detect object or other things */}).filter(frame);
```
# filters
```
    BRANCH(threshold);
    BRANCH(morphology);
    BRANCH(medianBlur);
    BRANCH(GaussianBlur);
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
```
