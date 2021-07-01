'''Program to clean the edges of an image. 

Cleans the edges of an input image, typically a drizzled Hubble Space Telescope (HST) image or any FITS 
image with a data and companion weight/error array. Replaces the poor quality data at the edges with 
sigma-clipped Gaussian noise based on the image pixel value distribution. This can be helpful for image 
alignment (e.g. using tweakreg from DrizzlePac https://www.stsci.edu/scientific-community/software/drizzlepac.html) 
or source detection (e.g. SExtractor/photutils) to prevent bad quality data at the image edges being 
flagged as sources. Also works for images with chip gaps. 

Main function is cleanedges_general, see below for calling sequence.

Written by Marc Rafelski. Translated to Python and adapted by Laura Prichard, Feb 2019, updated May 2020.
'''

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Gaussian2DKernel
import pylab as plt
import sys
from pdb import set_trace as st

def gaussian(x, h, mean, std):
	'''Gaussian function.

	Parameters
    ----------
    x : arr
    	Linear data array to fit with Gaussian.
    h : float 
    	Gaussian peak height.
    mean : float
    	Center of Gaussian peak.
    std : float
    	standard deviation/width of Gaussian.

    Returns
    -------
    y : arr
    	Gaussian y values (with dimensions of x)
	'''
	return h * np.exp(-((x - mean)**2 / (2 * std**2)))

def gauss_params(data, sig_clip=5.):
	'''Defines outlier-resistant Gaussian parameters (average and standard deviation) 
	for a 1D array, sigma-clips outliers with user-specified sigma, and derives new Gaussian 
	parameters for the clipped distribution. Similar to IDL's `histogauss` routine.

	Parameters
    ----------
    data : arr
    	Linear array of data values taken from the input images.
    sig_clip : float
    	User-defined sigma used for data clipping (default 5.).

    Returns
    -------
    data_clip : arr
    	Outlier-clipped data (same dimensions as `data`).
    new_gparams : arr
    	Outlier-resistant average and standard deviation for the clipped data.
	'''

	# Define arrays for Gauss parameters for before (gparams) and after sigma clipping (new_gparams)
	gparams = np.zeros([2])     
	new_gparams = np.zeros_like(gparams)  

	# Calculating outlier-resistant average [0] and sigma [1] of the distribution
	gparams[0] = np.nanmedian(data)               # Median, excluding NANs
	gparams[1] = mad_std(data, ignore_nan=True)   # MAD converted to a STD, excluding NANs

	# Set boundaries on the data around the average for clipping with user-defined sigma
	lo_lim = gparams[0] - sig_clip*gparams[1]
	hi_lim = gparams[0] + sig_clip*gparams[1]

	#Make unlinked copy of data to clip
	data_clip = np.copy(data)				

	# For all data outside the user-defined boundaries, set to the defined upper and lower sigma limits
	q = np.where(data_clip < lo_lim)[0]       # Find outlying data
	if q.shape[0] > 0: data_clip[q]=lo_lim    # If outlying data exists, clip it
	q = np.where(data_clip > hi_lim)[0]
	if q.shape[0] > 0: data_clip[q]=hi_lim

	# Calculating new outlier-resistant average [0] and sigma [1] of the clipped distribution
	new_gparams[0] = np.nanmedian(data_clip)               # Median, excluding NANs
	new_gparams[1] = mad_std(data_clip, ignore_nan=True)   # MAD converted to a STD, excluding NANs

	return data_clip, new_gparams


def cleanedges_general(img, build=True, wht_img=None, check=False, quiet=False, plot=False, kernel_std=13./2.335, kernel_size=41, sig_clip=5., sig_noise=3., ext_name='cln'):
	'''Main function for cleanedges_general. 
	Takes an input image file and replaces the edges with Gaussian noise.

	Parameters
    ----------
    img : str
    	Path to the input FITS image file to be cleaned. If build=False, this should be the path to the science 
    	image (e.g. '*_drc_sci.fits' if produced using AstroDrizzle with build=False).
	build : bool
		``True`` if the science and weight images are in one file with 'SCI' and 'WHT' extensions respectively
		(e.g. as produced by AstroDrizzle if build=True). ``False`` if img is just the path to the science FITS image
		(e.g. if produced using AstroDrizzle with build=False) in which case wht_img must be set to the path for the 
		corresponding weight/error image. Default ``True``.
	wht_img : str
		Path to the corresponding weight/error FITS file to the primary input science image (img). If build=False, this 
		must be set (e.g. '*_drc_wht.fits' if produced using AstroDrizzle with build=False). 
		e.g. cleanedges_general(img, build=False, wht_img=wht_img)
    check : bool
    	``True`` saves the kernel filtered/convolved image to the data directory as `img` with a '_check_conv.fits' 
    	extension. Default ``False``.
    quiet : bool
    	``True`` turns off code progress printed to terminal. Default ``False``.
    plot : bool
    	``True`` plots the histogram of clipped data from which Gaussian noise values are determined. Default ```False`.
    kernel_std : float
    	Standard deviation (STD) of the Gaussian kernel (in pixels) to smooth a mask of the data by to define where the edge 
    	regions to be cleaned are. Higher value means a thicker edge region will be cleaned. Default 13./2.335 ~ 5.5 sigma, 
    	where 13. is the FWHM and 2.335 is the conversion to a STD.
    kernel_size : int
    	Size in pixels of one edge of the smoothing Gaussian kernel area, by default x_size=y_size. Kernel must have odd 
    	dimensions so kernel_size must be an odd number. Default 41.
    sig_clip : float
    	Value to sigma-clip the data by to get initial guesses of the Gaussian distribution of pixel values in the input image. 
    	Default 5.
    sig_noise : float
    	The sigma-clipping of the noise to apply to the edges in order to clean them. Default 3.
    ext_name : str
    	String filename extension added to output cleaned images. Default 'cln'.
	'''

	# Printing to terminal
	if quiet==False: print('============================================')
	if quiet==False: print('Running cleanedges_general')
	if quiet==False: print('Cleaning edges of {}'.format(img)) 
	
	# Read in the science and weight images and their headers 
	if build==True:
		# If the data is combined into one file with different extensions (SCI and WHT)
		hdul = fits.open(img)
		drz, hdr1 = hdul['SCI'].data, hdul['SCI'].header   # Science extension
		wht, hdr2 = hdul['WHT'].data, hdul['WHT'].header   # Weight extension
	else:
		# If the science and weight images are separate files
		hdul = fits.open(img)
		drz, hdr1 = hdul[0].data, hdul[0].header   # Science image 
		# Check to see if the weight image path has been set
		try:
			hdu_wht = fits.open(wht_img)
		except ValueError:
			print('ERROR: If build=False, wht_img must be set to the file path of the weight image for the input science image.')
			raise
		if quiet==False: print('Using weight image {}'.format(wht_img)) 
		wht, hdr2 = hdu_wht[0].data, hdu_wht[0].header   # Weight image 

	# Determine noise properties for the data (i.e. where wht>0)
	datcut = np.where(wht > 0)
	dat = drz[datcut]             # Creates a 1D array of all the 'data' pixels in the image

	# Sigma (user-defined) clips data pixels to remove outliers (dat_clip) and get clipped data Gaussian parameters (new_gparams)
	if quiet==False: print('Sigma-clipping data ({}-sigma)'.format(sig_clip)) 
	dat_clip, new_gparams = gauss_params(dat, sig_clip=sig_clip)
	
	#Determine number of bins from no. pixels, rounded to nearest 10
	if dat_clip.shape[0]>85000: nbins = int(round(dat_clip.shape[0]/85000, -1))   #For a standard HST image, will be in the 100s        
	else: nbins=50         # If there are less than 85000 pixels, set a fixed number of bins

	# Plots the histogram of the clipped data 
	if plot==True: n, bins, patches = plt.hist(dat_clip, bins=nbins) 
	# Calculate histogram from cropped data
	n, bins = np.histogram(dat_clip, bins=nbins)

	# Calculating central values of histogram bins for fitting
	bin_cent = (bins[0:-1]+bins[1:])/2

	# Iteratively fitting the clipped data for the best fit with curve_fit()
	# Calling gaussian() with initial guesses for peak height, outlier-resistant average and sigma from the clipped distn.
	if quiet==False: print('Fitting sigma-clipped data with Gaussian') 
	popt, pcov = curve_fit(gaussian, bin_cent, n, [np.max(n), new_gparams[0], new_gparams[1]])

	# Over-plotting the best-fit Gaussian curve onto the histogram
	if plot==True: 
		xvals = np.linspace(dat_clip.min(), dat_clip.max(), 100)
		yvals = gaussian(xvals, *popt)
		plt.plot(xvals, yvals, color='red')
		plt.show()
		print('*** Enter `c` to continue running program, `q` to quit ***')
		st()

	# Defining best fit Gaussian parameters from curve_fit output, popt[0] is height
	mean = popt[1]    # best-fit Gaussian mean
	if mean < 0: mean = 0
	sigma = popt[2]   # best-fit Gaussian standard deviation/sigma

	if quiet==False: print('Creating mask for data and Gaussian kernel')
	# Create a mask of the data (data=1, non-data=0) for finding the edge regions to be cleaned
	step1 = np.where(wht != 0) 					# Find all 'data' pixels using the weight map
	cln = np.copy(drz)							# Copy input science data without linking the arrays
	cln[step1] = 1 								# Set data=1
	nan = np.where(np.isfinite(cln) == False)   # Find any NANs
	cln[nan] = 0 								# Replace all NANs with 0

	# Making a kernel to smooth the data mask by
	kernel = Gaussian2DKernel(kernel_std, x_size=kernel_size)

	# Create a filter by convolving the mask and kernel to create "blurred" edges, everything but 0s and 1s will be cleaned
	if quiet==False: print('Convolving mask with kernel') 
	filt = convolve(cln, kernel, boundary='fill', normalize_kernel=True) 
	# BOUNDARY OPTIONS: None, 'fill', 'wrap', 'extend'. In IDL: /edge_zero is same as 'fill' and fill_value=0 (default)
	
	# If check==True, user can check convolved output image
	if check==True:
		# Define output filename for the convolved image as the input file (img) with a '_check_conv' extension
		check_name = img.replace('.fits', '_check_conv.fits')
		fits.writeto(check_name, filt, header=hdr1, overwrite=True)
		print('Convolved image saved to {}'.format(check_name))
		print('Enter `c` to continue running program, `q` to quit, and then check=False to run program straight through')
		st()

	# Replace the input image edges with random noise in the image but limiting to 3 sigma
	bad2d = np.where((filt != 0) & (filt != 1))   #Returns two arrays, one of x and one of y coordinates for the bad pixels

	# Set random seed for repeatable "random" results
    	random.seed(16483)
	
	# Create array of random numbers with mean of 0 and sigma of 1
	ranarr = np.random.normal(loc=0.0, scale=1.0, size=bad2d[0].shape[0])

	if quiet==False: print('Replacing edges with random {}-sigma noise scaled to image params'.format(sig_noise)) 
	# Sigma-clipping the random noise by replacing outliers with another random number from a normal distribution
	ranarr[(ranarr>sig_noise)|(ranarr<-sig_noise)] = np.random.normal(loc=0.0, scale=1.0)  # Usually just this removes all outliers
	ranarr[(ranarr>sig_noise)|(ranarr<-sig_noise)] = np.random.normal(loc=0.0, scale=1.0)  # Do the second one to be thorough

	# Make a copy of drz and replace bad pixels with random Gaussian noise based on the image pixel distribution (mean, sigma)
	drz_cp = np.copy(drz)
	drz_cp[bad2d] = mean + (ranarr*sigma)

	# Updating the original with the cleaned image in the HDU List
	if build==True:
		# If the data is combined into one file with different extensions (SCI and WHT)
		hdul['SCI'].data = drz_cp
	else:
		# If the science and weight images are separate files
		hdul[0].data = drz_cp

	# Getting filename and appending the "_cln" (or user-specified ext_name) before the .fits extension
	out_img = img.replace('.fits', '_{}.fits'.format(ext_name))
	if quiet==False: print('Saving cleaned image to {}'.format(out_img)) 
	hdul.writeto(out_img, overwrite=True)
	if quiet==False: print('============================================')

