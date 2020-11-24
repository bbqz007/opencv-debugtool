```
opencv_createsamples -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec
Info file name: pos.txt
Img file name: (NULL)
Vec file name: pos.vec
BG  file name: (NULL)
Num: 1000
BG color: 0
BG threshold: 80
Invert: FALSE
Max intensity deviation: 40
Max x angle: 1.1
Max y angle: 1.1
Max z angle: 0.5
Show samples: FALSE
Width: 24
Height: 24
Max Scale: -1
RNG Seed: 12345
Create training samples from images collection...
pos.txt(32) : parse errorDone. Created 86 samples

opencv_traincascade -data cascade/ -vec pos.vec -bg neg.txt -precalcValBufSize 6000 -precalcIdxBufSize 6000 -numPos 86 -numNeg 86 -numStages 12 -w 24 -h 24 -maxFalseAlarmRate 0.4 -minHitRate 0.999
---------------------------------------------------------------------------------
Training parameters are pre-loaded from the parameter file in data folder!
Please empty this folder if you want to use a NEW set of training parameters.
---------------------------------------------------------------------------------
PARAMETERS:
cascadeDirName: cascade/
vecFileName: pos.vec
bgFileName: neg.txt
numPos: 86
numNeg: 86
numStages: 12
precalcValBufSize[Mb] : 6000
precalcIdxBufSize[Mb] : 6000
acceptanceRatioBreakValue : -1
stageType: BOOST
featureType: HAAR
sampleWidth: 24
sampleHeight: 24
boostType: GAB
minHitRate: 0.999
maxFalseAlarmRate: 0.4
weightTrimRate: 0.95
maxDepth: 1
maxWeakCount: 100
mode: BASIC
Number of unique features given windowSize [24,24] : 162336

Stages 0-11 are loaded
[root@localhost data]# rm cascade/*
rm: remove regular file 'cascade/cascade.xml'? q
rm: remove regular file 'cascade/params.xml'? ^C
[root@localhost data]# rm -f cascade/*
[root@localhost data]# /opt/z/tmp/opencv-3.4/build/bin/opencv_traincascade -data cascade/ -vec pos.vec -bg neg.txt -precalcValBufSize 6000 -precalcIdxBufSize 6000 -numPos 86 -numNeg 86 -numStages 12 -w 24 -h 24 -maxFalseAlarmRate 0.4 -minHitRate 0.999
PARAMETERS:
cascadeDirName: cascade/
vecFileName: pos.vec
bgFileName: neg.txt
numPos: 86
numNeg: 86
numStages: 12
precalcValBufSize[Mb] : 6000
precalcIdxBufSize[Mb] : 6000
acceptanceRatioBreakValue : -1
stageType: BOOST
featureType: HAAR
sampleWidth: 24
sampleHeight: 24
boostType: GAB
minHitRate: 0.999
maxFalseAlarmRate: 0.4
weightTrimRate: 0.95
maxDepth: 1
maxWeakCount: 100
mode: BASIC
Number of unique features given windowSize [24,24] : 162336

===== TRAINING 0-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 1
Precalculation time: 0
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1| 0.418605|
+----+---------+---------+
|   5|        1| 0.197674|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 4 seconds.

===== TRAINING 1-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 0.438776
Precalculation time: 0
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1| 0.383721|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 6 seconds.

===== TRAINING 2-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 0.20331
Precalculation time: 0
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1| 0.534884|
+----+---------+---------+
|   5|        1|0.0697674|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 9 seconds.

===== TRAINING 3-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 0.0400932
Precalculation time: 1
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1| 0.430233|
+----+---------+---------+
|   5|        1| 0.104651|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 13 seconds.

===== TRAINING 4-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 0.0196796
Precalculation time: 0
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|      0.5|
+----+---------+---------+
|   3|        1| 0.523256|
+----+---------+---------+
|   4|        1|0.0465116|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 16 seconds.

===== TRAINING 5-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 0.00598511
Precalculation time: 0
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1| 0.616279|
+----+---------+---------+
|   4|        1| 0.616279|
+----+---------+---------+
|   5|        1| 0.232558|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 20 seconds.

===== TRAINING 6-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 0.00507674
Precalculation time: 0
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1| 0.372093|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 22 seconds.

===== TRAINING 7-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 0.00282236
Precalculation time: 0
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1| 0.604651|
+----+---------+---------+
|   4|        1| 0.604651|
+----+---------+---------+
|   5|        1| 0.406977|
+----+---------+---------+
|   6|        1|0.0813954|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 27 seconds.

===== TRAINING 8-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 0.0011407
Precalculation time: 1
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1| 0.651163|
+----+---------+---------+
|   4|        1| 0.546512|
+----+---------+---------+
|   5|        1| 0.325581|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 32 seconds.

===== TRAINING 9-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 0.000908179
Precalculation time: 1
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|  0.55814|
+----+---------+---------+
|   5|        1| 0.639535|
+----+---------+---------+
|   6|        1|  0.22093|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 37 seconds.

===== TRAINING 10-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 0.000382714
Precalculation time: 1
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|  0.44186|
+----+---------+---------+
|   5|        1| 0.546512|
+----+---------+---------+
|   6|        1| 0.116279|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 44 seconds.

===== TRAINING 11-stage =====
<BEGIN
POS count : consumed   86 : 86
NEG count : acceptanceRatio    86 : 0.00014866
Precalculation time: 0
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1| 0.465116|
+----+---------+---------+
|   5|        1| 0.267442|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 54 seconds.

```

```
opencv_createsamples -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec
Info file name: pos.txt
Img file name: (NULL)
Vec file name: pos.vec
BG  file name: (NULL)
Num: 1000
BG color: 0
BG threshold: 80
Invert: FALSE
Max intensity deviation: 40
Max x angle: 1.1
Max y angle: 1.1
Max z angle: 0.5
Show samples: FALSE
Width: 24
Height: 24
Max Scale: -1
RNG Seed: 12345
Create training samples from images collection...
pos.txt(122) : parse errorDone. Created 343 samples

opencv_traincascade -data cascade/ -vec pos.vec -bg neg.txt -precalcValBufSize 6000 -precalcIdxBufSize 6000 -numPos 343 -numNeg 1000 -numStages 12 -w 24 -h 24 -maxFalseAlarmRate 0.4 -minHitRate 0.999
PARAMETERS:
cascadeDirName: cascade/
vecFileName: pos.vec
bgFileName: neg.txt
numPos: 343
numNeg: 1000
numStages: 12
precalcValBufSize[Mb] : 6000
precalcIdxBufSize[Mb] : 6000
acceptanceRatioBreakValue : -1
stageType: BOOST
featureType: HAAR
sampleWidth: 24
sampleHeight: 24
boostType: GAB
minHitRate: 0.999
maxFalseAlarmRate: 0.4
weightTrimRate: 0.95
maxDepth: 1
maxWeakCount: 100
mode: BASIC
Number of unique features given windowSize [24,24] : 162336

===== TRAINING 0-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 1
Precalculation time: 12
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|     0.68|
+----+---------+---------+
|   5|        1|    0.589|
+----+---------+---------+
|   6|        1|    0.285|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 51 seconds.

===== TRAINING 1-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 0.334225
Precalculation time: 13
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|        1|
+----+---------+---------+
|   5|        1|        1|
+----+---------+---------+
|   6|        1|    0.448|
+----+---------+---------+
|   7|        1|     0.48|
+----+---------+---------+
|   8|        1|     0.48|
+----+---------+---------+
|   9|        1|    0.494|
+----+---------+---------+
|  10|        1|    0.273|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 2 minutes 17 seconds.

===== TRAINING 2-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 0.113598
Precalculation time: 12
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|        1|
+----+---------+---------+
|   5|        1|        1|
+----+---------+---------+
|   6|        1|    0.673|
+----+---------+---------+
|   7|        1|    0.517|
+----+---------+---------+
|   8|        1|    0.448|
+----+---------+---------+
|   9|        1|    0.263|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 3 minutes 23 seconds.

===== TRAINING 3-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 0.038619
Precalculation time: 12
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|        1|
+----+---------+---------+
|   5|        1|    0.977|
+----+---------+---------+
|   6|        1|    0.818|
+----+---------+---------+
|   7|        1|    0.603|
+----+---------+---------+
|   8|        1|    0.643|
+----+---------+---------+
|   9|        1|    0.686|
+----+---------+---------+
|  10|        1|    0.452|
+----+---------+---------+
|  11|        1|     0.51|
+----+---------+---------+
|  12|        1|    0.545|
+----+---------+---------+
|  13|        1|    0.418|
+----+---------+---------+
|  14|        1|    0.215|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 5 minutes 9 seconds.

===== TRAINING 4-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 0.0144942
Precalculation time: 12
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|        1|
+----+---------+---------+
|   5|        1|    0.819|
+----+---------+---------+
|   6|        1|    0.871|
+----+---------+---------+
|   7|        1|    0.891|
+----+---------+---------+
|   8|        1|    0.804|
+----+---------+---------+
|   9|        1|    0.706|
+----+---------+---------+
|  10|        1|    0.687|
+----+---------+---------+
|  11|        1|    0.479|
+----+---------+---------+
|  12|        1|    0.563|
+----+---------+---------+
|  13|        1|     0.58|
+----+---------+---------+
|  14|        1|    0.387|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 6 minutes 46 seconds.

===== TRAINING 5-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 0.00594785
Precalculation time: 13
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|    0.908|
+----+---------+---------+
|   4|        1|    0.854|
+----+---------+---------+
|   5|        1|    0.866|
+----+---------+---------+
|   6|        1|     0.88|
+----+---------+---------+
|   7|        1|    0.903|
+----+---------+---------+
|   8|        1|    0.761|
+----+---------+---------+
|   9|        1|    0.625|
+----+---------+---------+
|  10|        1|    0.497|
+----+---------+---------+
|  11|        1|    0.449|
+----+---------+---------+
|  12|        1|    0.483|
+----+---------+---------+
|  13|        1|    0.383|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 8 minutes 21 seconds.

===== TRAINING 6-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 0.00339526
Precalculation time: 13
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|    0.854|
+----+---------+---------+
|   4|        1|    0.941|
+----+---------+---------+
|   5|        1|    0.966|
+----+---------+---------+
|   6|        1|    0.864|
+----+---------+---------+
|   7|        1|    0.696|
+----+---------+---------+
|   8|        1|    0.725|
+----+---------+---------+
|   9|        1|    0.657|
+----+---------+---------+
|  10|        1|    0.557|
+----+---------+---------+
|  11|        1|    0.489|
+----+---------+---------+
|  12|        1|    0.541|
+----+---------+---------+
|  13|        1|    0.502|
+----+---------+---------+
|  14|        1|    0.545|
+----+---------+---------+
|  15|        1|    0.546|
+----+---------+---------+
|  16|        1|    0.455|
+----+---------+---------+
|  17|        1|    0.456|
+----+---------+---------+
|  18|        1|     0.53|
+----+---------+---------+
|  19|        1|    0.387|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 10 minutes 34 seconds.

===== TRAINING 7-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 0.00169522
Precalculation time: 13
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|        1|
+----+---------+---------+
|   5|        1|    0.854|
+----+---------+---------+
|   6|        1|    0.877|
+----+---------+---------+
|   7|        1|    0.898|
+----+---------+---------+
|   8|        1|    0.919|
+----+---------+---------+
|   9|        1|    0.944|
+----+---------+---------+
|  10|        1|    0.717|
+----+---------+---------+
|  11|        1|    0.785|
+----+---------+---------+
|  12|        1|    0.725|
+----+---------+---------+
|  13|        1|    0.551|
+----+---------+---------+
|  14|        1|    0.499|
+----+---------+---------+
|  15|        1|    0.532|
+----+---------+---------+
|  16|        1|    0.332|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 12 minutes 33 seconds.

===== TRAINING 8-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 0.000666946
Precalculation time: 13
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|     0.88|
+----+---------+---------+
|   4|        1|    0.884|
+----+---------+---------+
|   5|        1|    0.885|
+----+---------+---------+
|   6|        1|    0.899|
+----+---------+---------+
|   7|        1|     0.91|
+----+---------+---------+
|   8|        1|    0.913|
+----+---------+---------+
|   9|        1|    0.846|
+----+---------+---------+
|  10|        1|    0.867|
+----+---------+---------+
|  11|        1|    0.804|
+----+---------+---------+
|  12|        1|    0.631|
+----+---------+---------+
|  13|        1|      0.6|
+----+---------+---------+
|  14|        1|    0.602|
+----+---------+---------+
|  15|        1|    0.484|
+----+---------+---------+
|  16|        1|    0.343|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 14 minutes 49 seconds.

===== TRAINING 9-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 0.000461582
Precalculation time: 13
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|        1|
+----+---------+---------+
|   5|        1|        1|
+----+---------+---------+
|   6|        1|    0.939|
+----+---------+---------+
|   7|        1|    0.797|
+----+---------+---------+
|   8|        1|    0.754|
+----+---------+---------+
|   9|        1|    0.816|
+----+---------+---------+
|  10|        1|    0.765|
+----+---------+---------+
|  11|        1|    0.792|
+----+---------+---------+
|  12|        1|    0.798|
+----+---------+---------+
|  13|        1|    0.638|
+----+---------+---------+
|  14|        1|    0.659|
+----+---------+---------+
|  15|        1|    0.508|
+----+---------+---------+
|  16|        1|    0.542|
+----+---------+---------+
|  17|        1|      0.5|
+----+---------+---------+
|  18|        1|    0.451|
+----+---------+---------+
|  19|        1|    0.344|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 17 minutes 21 seconds.

===== TRAINING 10-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 0.000243064
Precalculation time: 13
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|    0.966|
+----+---------+---------+
|   4|        1|    0.993|
+----+---------+---------+
|   5|        1|    0.993|
+----+---------+---------+
|   6|        1|    0.993|
+----+---------+---------+
|   7|        1|    0.945|
+----+---------+---------+
|   8|        1|    0.835|
+----+---------+---------+
|   9|        1|    0.861|
+----+---------+---------+
|  10|        1|    0.818|
+----+---------+---------+
|  11|        1|    0.764|
+----+---------+---------+
|  12|        1|      0.8|
+----+---------+---------+
|  13|        1|    0.801|
+----+---------+---------+
|  14|        1|    0.588|
+----+---------+---------+
|  15|        1|    0.626|
+----+---------+---------+
|  16|        1|    0.557|
+----+---------+---------+
|  17|        1|    0.511|
+----+---------+---------+
|  18|        1|    0.606|
+----+---------+---------+
|  19|        1|    0.685|
+----+---------+---------+
|  20|        1|    0.741|
+----+---------+---------+
|  21|        1|    0.379|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 20 minutes 39 seconds.

===== TRAINING 11-stage =====
<BEGIN
POS count : consumed   343 : 343
NEG count : acceptanceRatio    1000 : 0.000177017
Precalculation time: 13
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|     0.97|
+----+---------+---------+
|   4|        1|    0.974|
+----+---------+---------+
|   5|        1|    0.964|
+----+---------+---------+
|   6|        1|    0.964|
+----+---------+---------+
|   7|        1|    0.966|
+----+---------+---------+
|   8|        1|    0.972|
+----+---------+---------+
|   9|        1|    0.891|
+----+---------+---------+
|  10|        1|      0.8|
+----+---------+---------+
|  11|        1|    0.795|
+----+---------+---------+
|  12|        1|    0.825|
+----+---------+---------+
|  13|        1|    0.706|
+----+---------+---------+
|  14|        1|    0.689|
+----+---------+---------+
|  15|        1|    0.745|
+----+---------+---------+
|  16|        1|      0.6|
+----+---------+---------+
|  17|        1|    0.608|
+----+---------+---------+
|  18|        1|    0.436|
+----+---------+---------+
|  19|        1|    0.567|
+----+---------+---------+
|  20|        1|    0.483|
+----+---------+---------+
|  21|        1|    0.512|
+----+---------+---------+
|  22|        1|    0.457|
+----+---------+---------+
|  23|        1|    0.497|
+----+---------+---------+
|  24|        1|    0.321|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 24 minutes 25 seconds.

```
