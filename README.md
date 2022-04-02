# CSCM-Recognition
 # CSCM-Recognition
## Requirements
* Python 2.7
* PyTorch v0.4.1+
* pyclipper
* Polygon2
* OpenCV 3.4 (for c++ version pse)
* opencv-python 3.4


## Train

[**NOTE**] Some users say that they can't reproduce the reported performance with minor modification, like [1](https://github.com/ayumiymk/aster.pytorch/issues/17#issuecomment-527380815) and [2](https://github.com/ayumiymk/aster.pytorch/issues/17#issuecomment-528718596). I haven't try other settings, so I can't guarantee the same performance with different settings. The users should just run the following script without any modification to reproduce the results.
```
bash scripts/stn_att_rec.sh
```

## Test

You can test with .lmdb files by
```
bash scripts/main_test_all.sh
```
Or test with single image by
```
bash scripts/main_test_image.sh
```
