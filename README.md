# Douzero_Resnet_2.0
Douzero with ResNet and GPU support for Windows

Fixed bugs in the Resnet， 

use bigger network

add bidding system

auto_test.py: can be used to automatically test new models while training

SLModel: Threshold bidding Model Trained by Supervised Learning.
You can set up "Supervised" in evaluate.py to test the SLModel

The models in the best/ directory are the strongest Resnet models.
related project: [Douzero_Resnet](https://github.com/Vincentzyx/Douzero_Resnet)

The model in the test/ directory is the official ADP model. 
related project: [DouZero](https://github.com/kwai/DouZero) 
Paper: https://arxiv.org/abs/2106.06135

The bid folder holds 3 models for bidding:
[Douzero_resnet_bid](https://github.com/RuBP17/Douzero_resnet_bid) (LGPL license)


Contributor:

[EdwardPooh](https://github.com/EdwardPooh): Implements the resnet Model & Modify the training framework.

[Vincentzyx](https://github.com/Vincentzyx): Modify the training framework.

