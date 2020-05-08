# cleanedges
Program to clean the edges of an image.

Cleans the edges of an input image, typically a drizzled Hubble Space Telescope (HST) image or any FITS 
image with a data and companion weight/error array. Replaces the poor quality data at the edges with 
sigma-clipped Gaussian noise based on the image pixel value distribution. This can be helpful for image 
alignment (e.g. using tweakreg from DrizzlePac https://www.stsci.edu/scientific-community/software/drizzlepac.html) 
or source detection (e.g. SExtractor/photutils) to prevent bad quality data at the image edges being 
flagged as sources. Also works for images with chip gaps. Main function is cleanedges_general() in cleanedges_general.py, see there for input parameter descriptions.

Written by Marc Rafelski. Translated to Python and adapted by Laura Prichard, Feb 2019, updated May 2020.
