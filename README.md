# CUP-Net
## Matthew Parker 
## Honors Engineering Thesis Project
### Neural Network for CUP Imaging

Based on:
```
[BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1"](http://arxiv.org/abs/1602.02830),
Matthieu Courbariaux, Yoshua Bengio
```

Code based on: [Binary implementation in Keras](https://github.com/DingKe/nn_playground/tree/master/binarynet)


### Changes: 
In BinaryConv2D the kernel can be though of as a binary mask. That is, the kernel is simply multiplied (element-wise mult.) on the image rather than used in the traditioinal way.


In lambda_layers.py:

```python
def streak(input)

def integrate_ims(input)
```

The streak function is meant to simulate the streak camera and outputs a tensor of appropriate shape. 
The integrate function is the same function as 'keras.layers.add' but it accepts a single numpy array rather than a list of tensors.


