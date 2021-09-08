# DNNs_for_Optical_Flow
Summer project at HHMI Janelia. Using Pytorch to build DNNs that detect optical flow with biological plausibility in mind.


Presented as a poster at MIT URTC 2021 conference. Abstract as below:

Optical flow is the apparent motion of objects in a visual scene, computed between consecutive frames at a pixel level. Accurate and fast optical flow prediction is desirable in many use cases, such as object detection for self driving vehicles and robots, as well as camera motion estimation and various video editing applications. Optical flow is also an interesting problem in neuroscience, as human brains are constantly computing apparent motion to navigate the visual world. In this project, we investigated what features about the architecture of a deep neural net allow it to compute optical flow, findings from which we can use to inform the question of how the brain might compute the same problem. We find that a typical neural network architecture approach to optical flow, the image pyramid, may not actually improve upon a simple Unet architecture in terms of OF performance, and thusly not a good biological analog. We also note that greater statistical variations in training data allows for more robust model construction.
