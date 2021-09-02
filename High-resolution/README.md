# High Resolution

## Introduction

Our goal here is first to reduce the high-resolution image into a low-resolution image (basically we downscale a (2048, 2048, 3) array in a (512, 512, 3) array). Then we develop
an extension of the 32x32x3 compressor for images with shape (512, 512, 3). The low-resolution image will be compressed in 2 vectors of shape (32, 32, 128) and (256, 128). Let's 
summary what we are going to do : as input we get a 12 509 292 dimensional vector that we want to compress into a 163 840 dimensional vector. 

For the first step, we are going to use a RRDN structure :

<table>
  <tr>
    <td>High-resoltuion -> Low-resolution</td>
    <td><img src="images/HR-_LR.png"></td>
  </tr>
  
  <tr>
    <td>Low-resoltuion -> High-resolution</td>
    <td><img src="images/LR-_HR.png"></td>
  </tr>
</table>
