# Description

Conditional random filed (CRF) is a statistical modeling method applied in pattern recognition and machine learning.
It is recently used for processing the output of CNN semantic segmentation. 

This is a modified 3D version of a forked repository from https://github.com/cvlab-epfl/densecrf. 
The repository from https://github.com/cvlab-epfl/densecrf is a modified version of a forked [densecrf](http://www.philkr.net/home/densecrf), 
which was used as a part of the [DeepLab](https://bitbucket.org/deeplab/deeplab-public/).

For more details about the inference algorithm used in this version, please refer to and 
consider citing the following paper:
```
@article{baque2015principled,
  title={Principled Parallel Mean-Field Inference for Discrete Random Fields},
  author={Baqu{\'e}, Pierre and Bagautdinov, Timur and Fleuret, Fran{\c{c}}ois and Fua, Pascal},
  journal={arXiv preprint arXiv:1511.06103},
  year={2015}
}
```


# Building and Dependencies

You should have [matio](https://sourceforge.net/projects/matio/) library installed.

To build the binary, just run `make`.

# Usage

For the details (parameters) specific to this version, refer to 
`refine_pascal_nat/dense_inference.cpp`.


