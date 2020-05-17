# U-Net
U-Net for biomedical image segmentation.

## Dataset
The training data is a set of 30 sections from 
a serial section Transmission Electron Microscopy (ssTEM) dataset
 of the Drosophila first instar larva ventral nerve cord (VNC)
```
|——data
    |——train 
    |——rest 
```
The original dataset is from [Isbi challenge](http://brainiac2.mit.edu/isbi_challenge/), after downloaded files, zip the
directories like this path.

## Model
![picture/net-architecture.png](picture/net-architecture.png)
Input size was modified by [256, 256, 1]; 

At least convolution layer, data was activated by Sigmoid function to make sure pixels during [0,1] range.  