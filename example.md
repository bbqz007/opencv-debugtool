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
