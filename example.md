###
``` bash
cvtool images\apple-tree.awebp zoom,range,channel,morphology,morphology,contours
```
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
