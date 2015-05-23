"""
Tools for plotting / visualization
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Agg')  # no displayed figures -- need to call before loading pylab
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable

def is_square(shp, n_colors=1):
	"""
	Test whether entries in shp are square numbers, or are square numbers after divigind out the
	number of color channels.
	"""
	is_sqr = (shp == np.round(np.sqrt(shp))**2)
	is_sqr_colors = (shp == n_colors*np.round(np.sqrt(np.array(shp)/float(n_colors)))**2)
	return is_sqr | is_sqr_colors

def show_receptive_fields(theta, P=None, n_colors=None, max_display=100, grid_wa=None):
	"""
	Display receptive fields in a grid. Tries to intelligently guess whether to treat the rows,
	the columns, or the last two axes together as containing the receptive fields. It does this
	by checking which axes are square numbers -- so you can get some unexpected plots if the wrong
	axis is a square number, or if multiple axes are. It also tries to handle the last axis
	containing color channels correctly.
	"""

	shp = np.array(theta.shape)
	if n_colors is None:
		n_colors = 1
		if shp[-1] == 3:
			n_colors = 3
	# multiply colors in as appropriate
	if shp[-1] == n_colors:
		shp[-2] *= n_colors
		theta = theta.reshape(shp[:-1])
		shp = np.array(theta.shape)
	if len(shp) > 2:
		# merge last two axes
		shp[-2] *= shp[-1]
		theta = theta.reshape(shp[:-1])
		shp = np.array(theta.shape)
	if len(shp) > 2:
		# merge leading axes
		theta = theta.reshape((-1,shp[-1]))
		shp = np.array(theta.shape)
	if len(shp) == 1:
		theta = theta.reshape((-1,1))
		shp = np.array(theta.shape)

	# figure out the right orientation, by looking for the axis with a square
	# number of entries, up to number of colors. transpose if required
	is_sqr = is_square(shp, n_colors=n_colors)
	if is_sqr[0] and is_sqr[1]:
		warnings.warn("Unsure of correct matrix orientation. "
			"Assuming receptive fields along first dimension.")
	elif is_sqr[1]:
		theta = theta.T
	elif not is_sqr[0] and not is_sqr[1]:
		# neither direction corresponds well to an image
		# NOTE if you delete this next line, the code will work. The rfs just won't look very
		# image like
		return False

	theta = theta[:,:max_display].copy()

	if P is None:
		img_w = int(np.ceil(np.sqrt(theta.shape[0]/float(n_colors))))
	else:
		img_w = int(np.ceil(np.sqrt(P.shape[0]/float(n_colors))))
	nf = theta.shape[1]
	if grid_wa is None:
		grid_wa = int(np.ceil(np.sqrt(float(nf))))
	grid_wb = int(np.ceil(nf / float(grid_wa)))

	if P is not None:
		theta = np.dot(P, theta)

	vmin = np.min(theta)
	vmax = np.max(theta)

	for jj in range(nf):
		plt.subplot(grid_wa, grid_wb, jj+1)
		ptch = np.zeros((n_colors*img_w**2,))
		ptch[:theta.shape[0]] = theta[:,jj]
		if n_colors==3:
			ptch = ptch.reshape((img_w, img_w, n_colors))
		else:
			ptch = ptch.reshape((img_w, img_w))
		ptch -= vmin
		ptch /= vmax-vmin
		plt.imshow(ptch, interpolation='nearest', cmap=cm.Greys_r )
		plt.axis('off')

	return True


def plot_parameter(theta_in, base_fname_part1, base_fname_part2="", title = ''):
	"""
	Save both a raw and receptive field style plot of the contents of theta_in.
	base_fname_part1 provides the mandatory root of the filename.
	"""

	theta = np.array(theta_in.copy()) # in case it was a scalar
	print "%s min %g median %g mean %g max %g shape"%(
		title, np.min(theta), np.median(theta), np.mean(theta), np.max(theta)), theta.shape
	theta = np.squeeze(theta)
	if len(theta.shape) == 0:
		# it's a scalar -- make it a 1d array
		theta = np.array([theta])
	shp = theta.shape
	if len(shp) > 2:
		theta = theta.reshape((theta.shape[0], -1))
		shp = theta.shape

	## display basic figure
	plt.figure(figsize=[8,8])
	if len(shp) == 1:
		plt.plot(theta, '.', alpha=0.5)
	elif len(shp) == 2:
		plt.imshow(theta, interpolation='nearest', aspect='auto', cmap=cm.Greys_r)
		plt.colorbar()

	plt.title(title)
	plt.savefig(base_fname_part1 + '_raw_' + base_fname_part2 + '.pdf')
	plt.close()

	## also display it in basis function view if it's a matrix, or
	## if it's a bias with a square number of entries
	if len(shp) >= 2 or is_square(shp[0]):
		if len(shp) == 1:
			theta = theta.reshape((-1,1))
		plt.figure(figsize=[8,8])
		if show_receptive_fields(theta):
			plt.suptitle(title + "receptive fields")
			plt.savefig(base_fname_part1 + '_rf_' + base_fname_part2 + '.pdf')
		plt.close()

def plot_images(X, fname):
	"""
	Plot images in a grid.
	X is expected to be a 4d tensor of dimensions [# images]x[# colors]x[height]x[width]
	"""
	## plot
	# move color to end
	Xcol = X.transpose((0,2,3,1)).reshape((X.shape[0],-1)).T
	plt.figure(figsize=[8,8])
	if show_receptive_fields(Xcol, n_colors=X.shape[1]):
		plt.savefig(fname + '.pdf')
	else:
		warnings.warn('Images unexpected shape.')
	plt.close()

	## save as a .npz file
	np.savez(fname + '.npz', X=X)
