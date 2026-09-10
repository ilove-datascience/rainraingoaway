"""Use a one-pixel neighbourhood for local rain, rejecting out-of-area locations."""
import math
from masking import lowerLat, upperLat, lowerLong, upperLong, lat_long_to_pixel


def in_coverage(latitude, longitude):
    return (math.isfinite(latitude) and math.isfinite(longitude)
            and lowerLat <= latitude <= upperLat and lowerLong <= longitude <= upperLong)


def local_rain(grid, latitude, longitude):
    if not in_coverage(latitude, longitude):
        raise ValueError('Location is outside radar coverage')
    height, width = grid.shape
    x, y = lat_long_to_pixel(latitude, longitude, width, height)
    y = height - 1 - y  # display grids have already been flipped
    # Cross-shaped radius of one pixel; diagonals are farther away and excluded.
    values = [float(grid[py, px]) for px, py in ((x,y),(x-1,y),(x+1,y),(x,y-1),(x,y+1))
              if 0 <= px < width and 0 <= py < height]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Invalid rain values')
    north_south = (upperLat - lowerLat) * 111_195 / max(height - 1, 1)
    east_west = (upperLong - lowerLong) * 111_195 * math.cos(math.radians(latitude)) / max(width - 1, 1)
    radius = int(round(max(north_south, east_west) / 50) * 50)
    return max(values), (x, y), radius
