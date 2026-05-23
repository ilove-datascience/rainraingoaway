import math

lowerLat = 1.156
upperLat = 1.475
lowerLong = 103.565
upperLong = 104.13

distanceLat = math.fabs(upperLat - lowerLat)
distanceLong = math.fabs(upperLong - lowerLong)


def lat_long_to_pixel(lat: float, long: float, width: int, height: int) -> tuple[int, int]:
	"""Map a latitude/longitude to the closest pixel in the radar image.

	The bounds are treated as the geographic extent of the image:
	- lowerLat / upperLat define the vertical range
	- lowerLong / upperLong define the horizontal range

	The result is clamped to the image bounds so out-of-range coordinates
	still return a valid pixel.
	"""

	if width <= 0 or height <= 0:
		raise ValueError("width and height must be positive integers")

	if distanceLong == 0 or distanceLat == 0:
		raise ValueError("image bounds must span a non-zero geographic range")

	x_ratio = (long - lowerLong) / distanceLong
	y_ratio = (upperLat - lat) / distanceLat

	x = round(x_ratio * (width - 1))
	y = round(y_ratio * (height - 1))

	x = max(0, min(width - 1, x))
	y = max(0, min(height - 1, y))

	return x, y
