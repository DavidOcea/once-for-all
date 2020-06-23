import cv2
import math
import numbers
import random
import collections
import numpy as np
import collections
import sys
import torch
import warnings

try:
	import accimage
except ImportError:
	accimage = None

if sys.version_info < (3, 3):
	Sequence = collections.Sequence
	Iterable = collections.Iterable
else:
	Sequence = collections.abc.Sequence
	Iterable = collections.abc.Iterable

_opencv_interpolation_to_str = {
	cv2.INTER_NEAREST: 'PIL.Image.NEAREST',
	cv2.INTER_LINEAR: 'PIL.Image.BILINEAR',
	cv2.INTER_CUBIC: 'PIL.Image.BICUBIC',
	cv2.INTER_LANCZOS4: 'PIL.Image.LANCZOS',
	cv2.INTER_AREA: 'PIL.Image.AREA',
}

__all__ = ['Compose', 'Resize', 'CenterCrop', 'ToTensor', 'Normalize', 'RandomResizedCrop',
           'normalize', 'RandomHorizontalFlip', 'ColorJitter', 'Lighting']


def _is_numpy_image(img):
	return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _is_tensor_image(img):
	return torch.is_tensor(img) and img.ndimension() == 3


class Lambda(object):
	"""Apply a user-defined lambda as a transform.
	Args:
		lambd (function): Lambda/function to be used for transform.
	"""

	def __init__(self, lambd):
		assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
		self.lambd = lambd

	def __call__(self, img):
		return self.lambd(img)

	def __repr__(self):
		return self.__class__.__name__ + '()'


class Lighting(object):
	"""Lighting noise(AlexNet - style PCA - based noise)"""

	def __init__(self, alphastd, eigval, eigvec):
		self.alphastd = alphastd
		self.eigval = torch.tensor(eigval)
		self.eigvec = torch.tensor(eigvec)


	def __call__(self, img):
		if self.alphastd == 0:
			return img

		alpha = img.new().resize_(3).normal_(0, self.alphastd)
		rgb = self.eigvec.type_as(img).clone() \
			.mul(alpha.view(1, 3).expand(3, 3)) \
			.mul(self.eigval.view(1, 3).expand(3, 3)) \
			.sum(1).squeeze()

		return img.add(rgb.view(3, 1, 1).expand_as(img))


class Compose(object):
	"""Composes several transforms together.
	Args:
		transforms (list of ``Transform`` objects): list of transforms to compose.
	Example:
		>>> transforms.Compose([
		>>>     transforms.CenterCrop(10),
		>>>     transforms.ToTensor(),
		>>> ])
	"""

	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, img):
		for t in self.transforms:
			img = t(img)
		return img

	def __repr__(self):
		format_string = self.__class__.__name__ + '('
		for t in self.transforms:
			format_string += '\n'
			format_string += '    {0}'.format(t)
		format_string += '\n)'
		return format_string


class Resize(object):
	"""Resize the input PIL Image to the given size.
	Args:
		size (sequence or int): Desired output size. If size is a sequence like
			(h, w), output size will be matched to this. If size is an int,
			smaller edge of the image will be matched to this number.
			i.e, if height > width, then image will be rescaled to
			(size * height / width, size)
		interpolation (int, optional): Desired interpolation. Default is
			``PIL.Image.BILINEAR``
	"""

	def __init__(self, size, interpolation=cv2.INTER_LINEAR):
		assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
		self.size = size
		self.interpolation = interpolation

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be scaled.
		Returns:
			PIL Image: Rescaled image.
		"""
		if len(self.size) == 2:
			if type(self.size) is not type((1,2)):
				self.size = tuple(self.size)
			img = cv2.resize(img, self.size, interpolation=self.interpolation)
		else:
			img, _ = scale(img, short_size=self.size, interp=self.interpolation)
		return img

	def __repr__(self):
		interpolate_str = _opencv_interpolation_to_str[self.interpolation]
		return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class CenterCrop(object):
	"""Crops the given PIL Image at the center.
	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
	"""

	def __init__(self, size):
		if isinstance(size, numbers.Number):
			self.size = int(size)
		else:
			self.size = size

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be cropped.
		Returns:
			PIL Image: Cropped image.
		"""
		return center_crop(img, self.size)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0})'.format(self.size)


class ToTensor(object):
	"""Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
	Converts a PIL Image or numpy.ndarray (H x W x C) in the range
	[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
	if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
	or if the numpy.ndarray has dtype = np.uint8
	In the other cases, tensors are returned without scaling.
	"""

	def __call__(self, pic):
		"""
		Args:
			pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
		Returns:
			Tensor: Converted image.
		"""
		return to_tensor(pic)

	def __repr__(self):
		return self.__class__.__name__ + '()'


class Normalize(object):
	"""Normalize a tensor image with mean and standard deviation.
	Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
	will normalize each channel of the input ``torch.*Tensor`` i.e.
	``input[channel] = (input[channel] - mean[channel]) / std[channel]``
	.. note::
		This transform acts out of place, i.e., it does not mutates the input tensor.
	Args:
		mean (sequence): Sequence of means for each channel.
		std (sequence): Sequence of standard deviations for each channel.
	"""

	def __init__(self, mean, std, inplace=False):
		self.mean = mean
		self.std = std
		self.inplace = inplace

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized Tensor image.
		"""
		return normalize(tensor, self.mean, self.std, self.inplace)

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomResizedCrop(object):
	"""Crop the given PIL Image to random size and aspect ratio.
	A crop of random size (default: of 0.08 to 1.0) of the original size and a random
	aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
	is finally resized to given size.
	This is popularly used to train the Inception networks.
	Args:
		size: expected output size of each edge
		scale: range of size of the origin size cropped
		ratio: range of aspect ratio of the origin aspect ratio cropped
		interpolation: Default: PIL.Image.BILINEAR
	"""

	def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_LINEAR):
		if isinstance(size, tuple):
			self.size = size
		else:
			self.size = (size, size)
		if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
			warnings.warn("range should be of kind (min, max)")

		self.interpolation = interpolation
		self.scale = scale
		self.ratio = ratio

	@staticmethod
	def get_params(img, scale, ratio):
		"""Get parameters for ``crop`` for a random sized crop.
		Args:
			img (PIL Image): Image to be cropped.
			scale (tuple): range of size of the origin size cropped
			ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
		Returns:
			tuple: params (i, j, h, w) to be passed to ``crop`` for a random
				sized crop.
		"""
		area = img.shape[0] * img.shape[1]

		for attempt in range(10):
			target_area = random.uniform(*scale) * area
			log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
			aspect_ratio = math.exp(random.uniform(*log_ratio))

			w = int(round(math.sqrt(target_area * aspect_ratio)))
			h = int(round(math.sqrt(target_area / aspect_ratio)))

			if w < img.shape[0] and h < img.shape[1]:
				i = random.randint(0, img.shape[1] - h)
				j = random.randint(0, img.shape[0] - w)
				return i, j, h, w

		# Fallback to central crop
		# print('Fallback to central crop')
		in_ratio = img.shape[0] / img.shape[1]
		if (in_ratio < min(ratio)):
			w = img.shape[0]
			h = w / min(ratio)
		elif (in_ratio > max(ratio)):
			h = img.shape[1]
			w = h * max(ratio)
		else:  # whole image
			w = img.shape[0]
			h = img.shape[1]
		i = (img.shape[1] - h) // 2
		j = (img.shape[0] - w) // 2
		return int(i), int(j), int(h), int(w)

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be cropped and resized.
		Returns:
			PIL Image: Randomly cropped and resized image.
		"""
		i, j, h, w = self.get_params(img, self.scale, self.ratio)
		return resized_crop(img, i, j, h, w, self.size, self.interpolation)

	def __repr__(self):
		interpolate_str = _opencv_interpolation_to_str[self.interpolation]
		format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
		format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
		format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
		format_string += ', interpolation={0})'.format(interpolate_str)
		return format_string


class RandomHorizontalFlip(object):
	"""Horizontally flip the given PIL Image randomly with a given probability.
	Args:
		p (float): probability of the image being flipped. Default value is 0.5
	"""

	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be flipped.
		Returns:
			PIL Image: Randomly flipped image.
		"""
		if random.random() < self.p:
			return cv2.flip(img, 1)
		return img

	def __repr__(self):
		return self.__class__.__name__ + '(p={})'.format(self.p)


class ColorJitter(object):
	"""Randomly change the brightness, contrast and saturation of an image.
	Args:
		brightness (float or tuple of float (min, max)): How much to jitter brightness.
			brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
			or the given [min, max]. Should be non negative numbers.
		contrast (float or tuple of float (min, max)): How much to jitter contrast.
			contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
			or the given [min, max]. Should be non negative numbers.
		saturation (float or tuple of float (min, max)): How much to jitter saturation.
			saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
			or the given [min, max]. Should be non negative numbers.
		hue (float or tuple of float (min, max)): How much to jitter hue.
			hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
			Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
	"""

	def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
		self.brightness = self._check_input(brightness, 'brightness')
		self.contrast = self._check_input(contrast, 'contrast')
		self.saturation = self._check_input(saturation, 'saturation')
		self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
		                             clip_first_on_zero=False)

	def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
		if isinstance(value, numbers.Number):
			if value < 0:
				raise ValueError("If {} is a single number, it must be non negative.".format(name))
			value = [center - value, center + value]
			if clip_first_on_zero:
				value[0] = max(value[0], 0)
		elif isinstance(value, (tuple, list)) and len(value) == 2:
			if not bound[0] <= value[0] <= value[1] <= bound[1]:
				raise ValueError("{} values should be between {}".format(name, bound))
		else:
			raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

		# if value is 0 or (1., 1.) for brightness/contrast/saturation
		# or (0., 0.) for hue, do nothing
		if value[0] == value[1] == center:
			value = None
		return value

	@staticmethod
	def get_params(brightness, contrast, saturation, hue):
		"""Get a randomized transform to be applied on image.
		Arguments are same as that of __init__.
		Returns:
			Transform which randomly adjusts brightness, contrast and
			saturation in a random order.
		"""
		transforms = []

		if brightness is not None:
			brightness_factor = random.uniform(brightness[0], brightness[1])
			transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

		if contrast is not None:
			contrast_factor = random.uniform(contrast[0], contrast[1])
			transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

		if saturation is not None:
			saturation_factor = random.uniform(saturation[0], saturation[1])
			transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

		if hue is not None:
			hue_factor = random.uniform(hue[0], hue[1])
			transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

		random.shuffle(transforms)
		transform = Compose(transforms)

		return transform

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Input image.
		Returns:
			PIL Image: Color jittered image.
		"""
		transform = self.get_params(self.brightness, self.contrast,
		                            self.saturation, self.hue)
		return transform(img)

	def __repr__(self):
		format_string = self.__class__.__name__ + '('
		format_string += 'brightness={0}'.format(self.brightness)
		format_string += ', contrast={0}'.format(self.contrast)
		format_string += ', saturation={0}'.format(self.saturation)
		format_string += ', hue={0})'.format(self.hue)
		return format_string


def adjust_hue(img, hue_factor):
	if not (-0.5 <= hue_factor <= 0.5):
		raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

	if not _is_numpy_image(img):
		raise TypeError('img should be opencv Image. Got {}'.format(type(img)))

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h = hsv[:, :, 0]

	np_h = np.array(h, dtype=np.uint8)
	# uint8 addition take cares of rotation across boundaries
	with np.errstate(over='ignore'):
		np_h += np.uint8(hue_factor * 255)

	hsv[:, :, 0] = np_h
	img_r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return img_r


def adjust_saturation(img, saturation_factor):
	if not _is_numpy_image(img):
		raise TypeError('img should be opencv Image. Got {}'.format(type(img)))

	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_blend = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
	img_r = cv_blend(img_blend, img, saturation_factor)
	return img_r


def adjust_contrast(img, contrast_factor):
	if not _is_numpy_image(img):
		raise TypeError('img should be opencv Image. Got {}'.format(type(img)))

	b, g, r, _ = (cv2.mean(img))
	mean = (b + g + r) / 3.0 + 0.5
	img_blend = np.ones(img.shape, dtype=np.uint8) * mean
	img_r = cv_blend(img_blend, img, contrast_factor)
	return img_r


def adjust_brightness(img, brightness_factor):
	if not _is_numpy_image(img):
		raise TypeError('img should be opencv Image. Got {}'.format(type(img)))

	img_blend = np.zeros(img.shape, dtype=type(img))
	img_r = cv_blend(img_blend, img, brightness_factor)
	return img_r


def cv_blend(img1, img2, alpha):
	if alpha == 0:
		return img1
	if alpha == 1:
		return img2
	if alpha > 0 and alpha < 1:
		return np.uint8(img1 * (1 - alpha) + img2 * alpha)
	else:
		img = img1 * (1 - alpha) + img2 * alpha
		return np.uint8(np.clip(img, 0, 255))


def normalize(tensor, mean, std, inplace=False):
	"""Normalize a tensor image with mean and standard deviation.

	.. note::
		This transform acts out of place by default, i.e., it does not mutates the input tensor.

	See :class:`~torchvision.transforms.Normalize` for more details.

	Args:
		tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		mean (sequence): Sequence of means for each channel.
		std (sequence): Sequence of standard deviations for each channely.

	Returns:
		Tensor: Normalized Tensor image.
	"""
	if not _is_tensor_image(tensor):
		raise TypeError('tensor is not a torch image.')

	if not inplace:
		tensor = tensor.clone()

	mean = torch.tensor(mean, dtype=torch.float32)
	std = torch.tensor(std, dtype=torch.float32)
	tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
	return tensor


def resized_crop(img, i, j, h, w, size, interpolation=cv2.INTER_LINEAR):
	"""Crop the given PIL Image and resize it to desired size.
	Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.
	Args:
		img (PIL Image): Image to be cropped.
		i (int): i in (i,j) i.e coordinates of the upper left corner
		j (int): j in (i,j) i.e coordinates of the upper left corner
		h (int): Height of the cropped image.
		w (int): Width of the cropped image.
		size (sequence or int): Desired output size. Same semantics as ``resize``.
		interpolation (int, optional): Desired interpolation. Default is
			``PIL.Image.BILINEAR``.
	Returns:
		PIL Image: Cropped image.
	"""
	assert _is_numpy_image(img), 'img should be numpy Image'
	img = crop(img, i, j, h, w)
	img = cv2.resize(img, size, interpolation=interpolation)
	return img


def crop(img, i, j, h, w):
	"""Crop the given PIL Image.
	Args:
		img (PIL Image): Image to be cropped.
		i (int): i in (i,j) i.e coordinates of the upper left corner.
		j (int): j in (i,j) i.e coordinates of the upper left corner.
		h (int): Height of the cropped image.
		w (int): Width of the cropped image.
	Returns:
		PIL Image: Cropped image.
	"""
	if not _is_numpy_image(img):
		raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

	# return img.crop((j, i, j + w, i + h))
	j_end = np.clip(j + w, 0, img.shape[0])
	i_end = np.clip(i + h, 0, img.shape[1])
	return img[j:j_end, i:i_end]


def to_tensor(pic):
	"""Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
	See ``ToTensor`` for more details.
	Args:
		pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
	Returns:
		Tensor: Converted image.
	"""
	if not _is_numpy_image(pic):
		raise TypeError('pic should be opencv Image or ndarray. Got {}'.format(type(pic)))

	if isinstance(pic, np.ndarray):
		# handle numpy array
		if pic.ndim == 2:
			pic = pic[:, :, None]

		img = torch.from_numpy(pic.transpose((2, 0, 1)))
		# backward compatibility
		if isinstance(img, torch.ByteTensor):
			return img.float().div(255)
		else:
			return img

	if accimage is not None and isinstance(pic, accimage.Image):
		nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
		pic.copyto(nppic)
		return torch.from_numpy(nppic)

	# handle PIL Image
	if pic.mode == 'I':
		img = torch.from_numpy(np.array(pic, np.int32, copy=False))
	elif pic.mode == 'I;16':
		img = torch.from_numpy(np.array(pic, np.int16, copy=False))
	elif pic.mode == 'F':
		img = torch.from_numpy(np.array(pic, np.float32, copy=False))
	elif pic.mode == '1':
		img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
	else:
		img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
	# PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
	if pic.mode == 'YCbCr':
		nchannel = 3
	elif pic.mode == 'I;16':
		nchannel = 1
	else:
		nchannel = len(pic.mode)
	img = img.view(pic.size[1], pic.size[0], nchannel)
	# put it from HWC to CHW format
	# yikes, this transpose takes 80% of the loading time/CPU
	img = img.transpose(0, 1).transpose(0, 2).contiguous()
	if isinstance(img, torch.ByteTensor):
		return img.float().div(255)
	else:
		return img


def bgr2rgb(im):
	rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	return rgb_im


def rgb2bgr(im):
	bgr_im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
	return bgr_im


def normalize_numpy(im, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), rgb=False):
	if rgb:
		r, g, b = cv2.split(im)
	else:
		b, g, r = cv2.split(im)
	norm_im = cv2.merge([(b - mean[0]) / std[0], (g - mean[1]) / std[1], (r - mean[2]) / std[2]])
	return norm_im


def scale(im, short_size=256, max_size=1e5, interp=cv2.INTER_LINEAR):
	""" support gray im; interp: cv2.INTER_LINEAR (default) or cv2.INTER_NEAREST; """
	im_size_min = np.min(list(im.shape)[0:2])
	im_size_max = np.max(list(im.shape)[0:2])
	scale_ratio = float(short_size) / float(im_size_min)
	if np.round(scale_ratio * im_size_max) > float(max_size):
		scale_ratio = float(max_size) / float(im_size_max)

	scale_im = cv2.resize(im, None, None, fx=scale_ratio, fy=scale_ratio, interpolation=interp)

	return scale_im, scale_ratio


def scale_by_max(im, long_size=512, interp=cv2.INTER_LINEAR):
	""" support gray im; interp: cv2.INTER_LINEAR (default) or cv2.INTER_NEAREST; """
	im_size_max = np.max(im.shape[0:2])
	scale_ratio = float(long_size) / float(im_size_max)

	scale_im = cv2.resize(im, None, None, fx=scale_ratio, fy=scale_ratio, interpolation=interp)

	return scale_im, scale_ratio


def scale_by_target(im, target_size=(512, 256), interp=cv2.INTER_LINEAR):
	""" target_size=(h, w), support gray im; interp: cv2.INTER_LINEAR (default) or cv2.INTER_NEAREST; """
	min_factor = min(float(target_size[0]) / float(im.shape[0]),
	                 float(target_size[1]) / float(im.shape[1]))

	scale_im = cv2.resize(im, None, None, fx=min_factor, fy=min_factor, interpolation=interp)

	return scale_im, min_factor


def rotate(im, degree=0, borderValue=(0, 0, 0), interp=cv2.INTER_LINEAR):
	""" support gray im; interp: cv2.INTER_LINEAR (default) or cv2.INTER_NEAREST; """
	h, w = im.shape[:2]
	rotate_mat = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)
	rotation = cv2.warpAffine(im, rotate_mat, (w, h), flags=interp,
	                          borderValue=cv2.cv.Scalar(borderValue[0], borderValue[1], borderValue[2]))

	return rotation


def HSV_adjust(im, color=1.0, contrast=1.0, brightness=1.0):
	_HSV = np.dot(cv2.cvtColor(im, cv2.COLOR_BGR2HSV).reshape((-1, 3)),
	              np.array([[color, 0, 0], [0, contrast, 0], [0, 0, brightness]]))

	_HSV_H = np.where(_HSV < 255, _HSV, 255)
	hsv = cv2.cvtColor(np.uint8(_HSV_H.reshape((-1, im.shape[1], 3))), cv2.COLOR_HSV2BGR)

	return hsv


def salt_pepper(im, SNR=1.0):
	""" SNR: better >= 0.9; """
	noise_num = int((1 - SNR) * im.shape[0] * im.shape[1])
	noise_im = im.copy()
	for i in range(noise_num):
		rand_x = np.random.random_integers(0, im.shape[0] - 1)
		rand_y = np.random.random_integers(0, im.shape[1] - 1)

		if np.random.random_integers(0, 1) == 0:
			noise_im[rand_x, rand_y] = 0
		else:
			noise_im[rand_x, rand_y] = 255

	return noise_im


def padding_im(im, target_size=(512, 512), borderType=cv2.BORDER_CONSTANT, mode=0):
	""" support gray im; target_size=(h, w); mode=0 left-top, mode=1 center; """
	if mode not in (0, 1):
		raise Exception("mode need to be one of 0 or 1, 0 for left-top mode, 1 for center mode.")

	pad_h_top = max(int((target_size[0] - im.shape[0]) * 0.5), 0) * mode
	pad_h_bottom = max(target_size[0] - im.shape[0], 0) - pad_h_top
	pad_w_left = max(int((target_size[1] - im.shape[1]) * 0.5), 0) * mode
	pad_w_right = max(target_size[1] - im.shape[1], 0) - pad_w_left

	if borderType == cv2.BORDER_CONSTANT:
		pad_im = cv2.copyMakeBorder(im, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, cv2.BORDER_CONSTANT)
	else:
		pad_im = cv2.copyMakeBorder(im, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, borderType)

	return pad_im


def extend_bbox(im, bbox, margin=(0.5, 0.5, 0.5, 0.5)):
	box_w = int(bbox[2] - bbox[0])
	box_h = int(bbox[3] - bbox[1])

	new_x1 = max(1, bbox[0] - margin[0] * box_w)
	new_y1 = max(1, bbox[1] - margin[1] * box_h)
	new_x2 = min(im.shape[1] - 1, bbox[2] + margin[2] * box_w)
	new_y2 = min(im.shape[0] - 1, bbox[3] + margin[3] * box_h)

	return np.asarray([new_x1, new_y1, new_x2, new_y2])


def bbox_crop(im, bbox):
	return im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]


def center_crop(im, crop_size=224):  # single crop
	im_shape = list(im.shape)
	im_size_min = min(im_shape[:2])
	if im_size_min < crop_size:
		return
	yy = int((im.shape[0] - crop_size) / 2)
	xx = int((im.shape[1] - crop_size) / 2)
	crop_im = im[yy: yy + crop_size, xx: xx + crop_size]

	return crop_im


def over_sample(im, crop_size=224):  # 5 crops of image
	im_size_min = min(im.shape[:2])
	if im_size_min < crop_size:
		return
	yy = int((im.shape[0] - crop_size) / 2)
	xx = int((im.shape[1] - crop_size) / 2)
	sample_list = [im[:crop_size, :crop_size], im[-crop_size:, -crop_size:], im[:crop_size, -crop_size:],
	               im[-crop_size:, :crop_size], im[yy: yy + crop_size, xx: xx + crop_size]]

	return sample_list


def mirror_crop(im, crop_size=224):  # 10 crops
	crop_list = []
	mirror = im[:, ::-1]
	crop_list.extend(over_sample(im, crop_size=crop_size))
	crop_list.extend(over_sample(mirror, crop_size=crop_size))

	return crop_list


def multiscale_mirrorcrop(im, scales=(256, 288, 320, 352)):  # 120(4*3*10) crops
	crop_list = []
	im_size_min = np.min(im.shape[0:2])
	for i in scales:
		resize_im = cv2.resize(im, (im.shape[1] * i / im_size_min, im.shape[0] * i / im_size_min))
		yy = int((resize_im.shape[0] - i) / 2)
		xx = int((resize_im.shape[1] - i) / 2)
		for j in range(3):
			left_center_right = resize_im[yy * j: yy * j + i, xx * j: xx * j + i]
			mirror = left_center_right[:, ::-1]
			crop_list.extend(over_sample(left_center_right))
			crop_list.extend(over_sample(mirror))

	return crop_list


def multi_scale(im, scales=(480, 576, 688, 864, 1200), max_sizes=(800, 1000, 1200, 1500, 1800), image_flip=False):
	im_size_min = np.min(im.shape[0:2])
	im_size_max = np.max(im.shape[0:2])

	scale_ims = []
	scale_ratios = []
	for i in range(len(scales)):
		scale_ratio = float(scales[i]) / float(im_size_min)
		if np.round(scale_ratio * im_size_max) > float(max_sizes[i]):
			scale_ratio = float(max_sizes[i]) / float(im_size_max)
		resize_im = cv2.resize(im, None, None, fx=scale_ratio, fy=scale_ratio,
		                       interpolation=cv2.INTER_LINEAR)
		scale_ims.append(resize_im)
		scale_ratios.append(scale_ratio)
		if image_flip:
			scale_ims.append(cv2.resize(im[:, ::-1], None, None, fx=scale_ratio, fy=scale_ratio,
			                            interpolation=cv2.INTER_LINEAR))
			scale_ratios.append(-scale_ratio)

	return scale_ims, scale_ratios


def multi_scale_by_max(im, scales=(480, 576, 688, 864, 1200), image_flip=False):
	im_size_max = np.max(im.shape[0:2])

	scale_ims = []
	scale_ratios = []
	for i in range(len(scales)):
		scale_ratio = float(scales[i]) / float(im_size_max)

		resize_im = cv2.resize(im, None, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)
		scale_ims.append(resize_im)
		scale_ratios.append(scale_ratio)
		if image_flip:
			scale_ims.append(cv2.resize(im[:, ::-1], None, None, fx=scale_ratio, fy=scale_ratio,
			                            interpolation=cv2.INTER_LINEAR))
			scale_ratios.append(-scale_ratio)

	return scale_ims, scale_ratios


def pil_resize(im, size):
	from PIL import Image
	interpolation = Image.BILINEAR
	if isinstance(size, int):
		w, h = im.size
		if (w <= h and w == size) or (h <= w and h == size):
			return im
		if w < h:
			ow = size
			oh = int(size * h / w)
			return im.resize((ow, oh), interpolation)
		else:
			oh = size
			ow = int(size * w / h)
			return im.resize((ow, oh), interpolation)
	else:
		return im.resize(size[::-1], interpolation)


def mask_kpt_resize(im, mask, kpt, center, ratio):
	"""Resize the ``numpy.ndarray`` and points as ratio.
	Args:
		im     (numpy.ndarray):   Image to be resized.
		mask   (numpy.ndarray):   Mask to be resized.
		kpt    (list):            Keypoints to be resized.
		center (list):            Center points to be resized.
		ratio  (tuple or number): the ratio to resize.
	Returns:
		numpy.ndarray: Resized image.
		numpy.ndarray: Resized mask.
		lists:         Resized keypoints.
		lists:         Resized center points.
	"""

	if not (isinstance(ratio, numbers.Number) or (isinstance(ratio, collections.Iterable) and len(ratio) == 2)):
		raise TypeError('Got inappropriate ratio arg: {}'.format(ratio))

	h, w, _ = im.shape
	if w < 64:
		im = cv2.copyMakeBorder(im, 0, 0, 0, 64 - w, cv2.BORDER_CONSTANT, value=(128, 128, 128))
		mask = cv2.copyMakeBorder(mask, 0, 0, 0, 64 - w, cv2.BORDER_CONSTANT, value=(1, 1, 1))
		w = 64

	if isinstance(ratio, numbers.Number):
		num = len(kpt)
		length = len(kpt[0])
		for i in range(num):
			for j in range(length):
				kpt[i][j][0] *= ratio
				kpt[i][j][1] *= ratio
			center[i][0] *= ratio
			center[i][1] *= ratio

		return cv2.resize(im, (0, 0), fx=ratio, fy=ratio), cv2.resize(mask, (0, 0), fx=ratio, fy=ratio), kpt, center

	else:
		num = len(kpt)
		length = len(kpt[0])
		for i in range(num):
			for j in range(length):
				kpt[i][j][0] *= ratio[0]
				kpt[i][j][1] *= ratio[1]
			center[i][0] *= ratio[0]
			center[i][1] *= ratio[1]
		return np.ascontiguousarray(cv2.resize(im, (0, 0), fx=ratio[0], fy=ratio[1])), np.ascontiguousarray(
			cv2.resize(mask, (0, 0), fx=ratio[0], fy=ratio[1])), kpt, center


def mask_kpt_rotate(im, mask, kpt, center, degree):
	"""Rotate the ``numpy.ndarray`` and points as degree.
	Args:
		im     (numpy.ndarray): Image to be rotated.
		mask   (numpy.ndarray): Mask to be rotated.
		kpt    (list):          Keypoints to be rotated.
		center (list):          Center points to be rotated.
		degree (number):        the degree to rotate.
	Returns:
		numpy.ndarray: Resized image.
		numpy.ndarray: Resized mask.
		list:          Resized keypoints.
		list:          Resized center points.
	"""

	height, width, _ = im.shape

	im_center = (width / 2.0, height / 2.0)

	rotateMat = cv2.getRotationMatrix2D(im_center, degree, 1.0)
	cos_val = np.abs(rotateMat[0, 0])
	sin_val = np.abs(rotateMat[0, 1])
	new_width = int(height * sin_val + width * cos_val)
	new_height = int(height * cos_val + width * sin_val)
	rotateMat[0, 2] += (new_width / 2.) - im_center[0]
	rotateMat[1, 2] += (new_height / 2.) - im_center[1]

	im = cv2.warpAffine(im, rotateMat, (new_width, new_height), borderValue=(128, 128, 128))
	mask = cv2.warpAffine(mask, rotateMat, (new_width, new_height), borderValue=(1, 1, 1))

	num = len(kpt)
	length = len(kpt[0])
	for i in range(num):
		for j in range(length):
			x = kpt[i][j][0]
			y = kpt[i][j][1]
			p = np.array([x, y, 1])
			p = rotateMat.dot(p)
			kpt[i][j][0] = p[0]
			kpt[i][j][1] = p[1]

		x = center[i][0]
		y = center[i][1]
		p = np.array([x, y, 1])
		p = rotateMat.dot(p)
		center[i][0] = p[0]
		center[i][1] = p[1]

	return np.ascontiguousarray(im), np.ascontiguousarray(mask), kpt, center


def mask_kpt_crop(im, mask, kpt, center, offset_left, offset_up, w, h):
	num = len(kpt)
	length = len(kpt[0])

	for x in range(num):
		for y in range(length):
			kpt[x][y][0] -= offset_left
			kpt[x][y][1] -= offset_up
		center[x][0] -= offset_left
		center[x][1] -= offset_up

	height, width, _ = im.shape
	mask = mask.reshape((height, width))

	new_im = np.empty((h, w, 3), dtype=np.float32)
	new_im.fill(128)

	new_mask = np.empty((h, w), dtype=np.float32)
	new_mask.fill(1)

	st_x = 0
	ed_x = w
	st_y = 0
	ed_y = h
	or_st_x = offset_left
	or_ed_x = offset_left + w
	or_st_y = offset_up
	or_ed_y = offset_up + h

	if offset_left < 0:
		st_x = -offset_left
		or_st_x = 0
	if offset_left + w > width:
		ed_x = width - offset_left
		or_ed_x = width
	if offset_up < 0:
		st_y = -offset_up
		or_st_y = 0
	if offset_up + h > height:
		ed_y = height - offset_up
		or_ed_y = height

	new_im[st_y: ed_y, st_x: ed_x, :] = im[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()
	new_mask[st_y: ed_y, st_x: ed_x] = mask[or_st_y: or_ed_y, or_st_x: or_ed_x].copy()

	return np.ascontiguousarray(new_im), np.ascontiguousarray(new_mask), kpt, center


def mask_kpt_hflip(im, mask, kpt, center):
	height, width, _ = im.shape
	mask = mask.reshape((height, width, 1))

	im = im[:, ::-1, :]
	mask = mask[:, ::-1, :]

	num = len(kpt)
	length = len(kpt[0])
	for i in range(num):
		for j in range(length):
			if kpt[i][j][2] <= 1:
				kpt[i][j][0] = width - 1 - kpt[i][j][0]
		center[i][0] = width - 1 - center[i][0]

	swap_pair = [[3, 6], [4, 7], [5, 8], [9, 12], [10, 13], [11, 14], [15, 16], [17, 18]]
	for x in swap_pair:
		for i in range(num):
			temp_point = kpt[i][x[0] - 1]
			kpt[i][x[0] - 1] = kpt[i][x[1] - 1]
			kpt[i][x[1] - 1] = temp_point

	return np.ascontiguousarray(im), np.ascontiguousarray(mask), kpt, center


class RandomPixelJitter(object):
	def __init__(self, pixel_range):
		self.pixel_range = pixel_range
		assert len(pixel_range) == 2

	def __call__(self, im):
		pic = np.array(im)
		noise = np.random.randint(self.pixel_range[0], self.pixel_range[1], pic.shape[-1])
		pic = pic + noise
		pic = pic.astype(np.uint8)
		return pic


class MasKptRandomResized(object):
	"""Resize the given numpy.ndarray to random size and aspect ratio.
	Args:
		scale_min: the min scale to resize.
		scale_max: the max scale to resize.
	"""

	def __init__(self, scale_min=0.5, scale_max=1.1):
		self.scale_min = scale_min
		self.scale_max = scale_max

	@staticmethod
	def get_params(im, scale_min, scale_max, scale):
		height, width, _ = im.shape

		ratio = random.uniform(scale_min, scale_max)
		ratio = ratio * 0.6 / scale

		return ratio

	def __call__(self, im, mask, kpt, center, scale):
		"""
		Args:
			im      (numpy.ndarray): Image to be resized.
			mask    (numpy.ndarray): Mask to be resized.
			kpt     (list):          keypoints to be resized.
			center: (list):          center points to be resized.
		Returns:
			numpy.ndarray: Randomly resize image.
			numpy.ndarray: Randomly resize mask.
			list:          Randomly resize keypoints.
			list:          Randomly resize center points.
		"""
		ratio = self.get_params(im, self.scale_min, self.scale_max, scale[0])

		return mask_kpt_resize(im, mask, kpt, center, ratio)


class MasKptResized(object):
	"""Resize the given numpy.ndarray to the size for test.
	Args:
		size: the size to resize.
	"""

	def __init__(self, size):
		assert (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2))
		if isinstance(size, int):
			self.size = (size, size)
		else:
			self.size = size

	@staticmethod
	def get_params(im, output_size):

		height, width, _ = im.shape

		return (output_size[0] * 1.0 / width, output_size[1] * 1.0 / height)

	def __call__(self, im, mask, kpt, center):
		"""
		Args:
			im      (numpy.ndarray): Image to be resized.
			mask    (numpy.ndarray): Mask to be resized.
			kpt     (list):          keypoints to be resized.
			center: (list):          center points to be resized.
		Returns:
			numpy.ndarray: Randomly resize image.
			numpy.ndarray: Randomly resize mask.
			list:          Randomly resize keypoints.
			list:          Randomly resize center points.
		"""
		ratio = self.get_params(im, self.size)

		return mask_kpt_resize(im, mask, kpt, center, ratio)


class MasKptRandomRotation(object):
	"""Rotate the input numpy.ndarray and points to the given degree.
	Args:
		degree_range (number): Desired rotate degree range.
	"""

	def __init__(self, degree_range):
		self.degree_range = degree_range
		assert len(degree_range) == 2

	@staticmethod
	def get_params(min_degree, max_degree):
		"""Get parameters for ``rotate`` for a random rotate.
		Returns:
			number: degree to be passed to ``rotate`` for random rotate.
		"""
		degree = random.uniform(min_degree, max_degree)

		return degree

	def __call__(self, im, mask, kpt, center):
		"""
		Args:
			im     (numpy.ndarray): Image to be rotated.
			mask   (numpy.ndarray): Mask to be rotated.
			kpt    (list):          Keypoints to be rotated.
			center (list):          Center points to be rotated.
		Returns:
			numpy.ndarray: Rotated image.
			list:          Rotated key points.
		"""
		degree = self.get_params(self.degree_range[0], self.degree_range[1])

		return mask_kpt_rotate(im, mask, kpt, center, degree)


class MasKptRandomCrop(object):
	"""Crop the given numpy.ndarray and  at a random location.
	Args:
		size (int): Desired output size of the crop.
	"""

	def __init__(self, size, center_perturb_max=40):
		assert isinstance(size, numbers.Number)
		self.size = (int(size), int(size))  # (w, h)
		self.center_perturb_max = center_perturb_max

	@staticmethod
	def get_params(im, center, output_size, center_perturb_max):
		"""Get parameters for ``crop`` for a random crop.
		Args:
			im                (numpy.ndarray): Image to be cropped.
			center             (list):          the center of main person.
			output_size        (tuple):         Expected output size of the crop.
			center_perturb_max (int):           the max perturb size.
		Returns:
			tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
		"""
		ratio_x = random.uniform(0, 1)
		ratio_y = random.uniform(0, 1)
		x_offset = int((ratio_x - 0.5) * 2 * center_perturb_max)
		y_offset = int((ratio_y - 0.5) * 2 * center_perturb_max)
		center_x = center[0][0] + x_offset
		center_y = center[0][1] + y_offset

		return int(round(center_x - output_size[0] / 2)), int(round(center_y - output_size[1] / 2))

	def __call__(self, im, mask, kpt, center):
		"""
		Args:
			im (numpy.ndarray): Image to be cropped.
			mask (numpy.ndarray): Mask to be cropped.
			kpt (list): keypoints to be cropped.
			center (list): center points to be cropped.
		Returns:
			numpy.ndarray: Cropped image.
			numpy.ndarray: Cropped mask.
			list:          Cropped keypoints.
			list:          Cropped center points.
		"""

		offset_left, offset_up = self.get_params(im, center, self.size, self.center_perturb_max)

		return mask_kpt_crop(im, mask, kpt, center, offset_left, offset_up, self.size[0], self.size[1])


class MasKptRandomHorizontalFlip(object):
	"""Random horizontal flip the image.
	Args:
		prob (number): the probability to flip.
	"""

	def __init__(self, prob=0.5):
		self.prob = prob

	def __call__(self, im, mask, kpt, center):
		"""
		Args:
			im     (numpy.ndarray): Image to be flipped.
			mask   (numpy.ndarray): Mask to be flipped.
			kpt    (list):          Keypoints to be flipped.
			center (list):          Center points to be flipped.
		Returns:
			numpy.ndarray: Randomly flipped image.
			list: Randomly flipped points.
		"""
		if random.random() < self.prob:
			return mask_kpt_hflip(im, mask, kpt, center)
		return im, mask, kpt, center


class MasKptCompose(object):
	"""Composes several transforms together.
	Args:
		transforms (list of ``Transform`` objects): list of transforms to compose.
	Example:
		>>> Mytransforms.Compose([
		>>>     Mytransforms.CenterCrop(10),
		>>>     Mytransforms.ToTensor(),
		>>> ])
	"""

	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, im, mask, kpt, center, scale=None):
		for t in self.transforms:
			if isinstance(t, MasKptRandomResized):
				im, mask, kpt, center = t(im, mask, kpt, center, scale)
			else:
				im, mask, kpt, center = t(im, mask, kpt, center)

		return im, mask, kpt, center
