###
``` bash
cvtool images\apple-tree.awebp zoom,range,channel,morphology,morphology,contours
```
![img](https://img2024.cnblogs.com/blog/665551/202503/665551-20250303231630247-935422039.png)
* zoom
  * x 50%
  * y 50% 
* range
  * hsv
  * h 0-10
  * s 43-255
  * v 46-255
  * mask only
* morphology
  * kernel on
  * morphology 10
  * method close
* morphology
  * kernel on
  * morphology 5
  * method open
* contours
  * show area 2000
  * show convex hull on         

``` bash
cvtool images\apple-tree01.awebp zoom,range,channel,morphology,morphology,contours
```
![img](https://img2024.cnblogs.com/blog/665551/202503/665551-20250304222907066-1352058251.png)
* zoom
  * x 50%
  * y 50% 
* range
  * hsv
  * h 10-118
  * s 0-255
  * v 0-255
  * invert mask
  * mask only
* morphology
  * kernel on
  * morphology 6
  * method close
* morphology
  * kernel on
  * morphology 5
  * method open
* contours
  * show area 1600
