# HSHT-Satellite-Imagery-Synthesis
Code used **Improving Flood Maps by Increasing the Temporal Resolution of Satellites Using Hybrid Sensor Fusion - Video Interpolation Networks**

### The work is heavily based on: 
* Pix2pix from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
* FLAVR from: https://github.com/tarun005/FLAVR

### Results

![](img/results.JPG)

### Rough Idea:
We propose the combination of video frame interpolation on a single satellite and cross satellite sensor fusion via image-to-image translation as a means of creating higher spatial higher temporal imagery. 

To do this, we first modified FLAVR and Pix2pix to handle satellite imagery. We then downloaded a custom dataset of MODIS and Landast 8 using [gee_downloader](https://github.com/yuvalofek/GEE_Downloader) to use as our data. 

We combined the two networks in a variety of ways as shown in in the following diagrams: 
1. Naive - averaging the outputs of the image translation and video interpolation models.

![](img/naive.JPG)

2. Secondary sensor-fusion - use the outputs of the image interpolation and satellite sensor fusion networks as inputs to a secondary sensor fusion network.

![](img/2nd_fusion.JPG)

3. High-resolution information boosting - feeding the output of the video interpolation to the image translation model to boost its understanding of the high-resolution space.

![](img/hr-boosting.JPG)

4. High-temporal information injection - injecting low-resolution high temporal information to the video interpolation model.

![](img/injflavr1.JPG)
![](img/injflavr2.JPG)

The model used to inject the low resolution images is dubbed injFLAVR and can be seen in the following diagram:

![](img/injflavr_arch.JPG)
