import math
from PIL import Image

intensityColors = [
    '#40FFFD', '#3BEEEC', '#32D0D2', '#2CB9BD', '#229698',
    '#1C827D', '#1B8742', '#229F44', '#27B240', '#2CC53B',
    '#30D43E', '#38EF46', '#3BFB49', '#59FA61', '#FEFB63',
    '#FDFA53', '#FDEB50', '#FDD74A', '#FCC344', '#FAB03F',
    '#FAA23D', '#FB8938', '#FB7133', '#F94C2D', '#F9282A',
    '#DD1423', '#BE0F1D', '#B21867', '#D028A6', '#F93DF5',
]

intensityColorsCount = len(intensityColors)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return {
        "r": int(hex_color[0:2], 16),
        "g": int(hex_color[2:4], 16),
        "b": int(hex_color[4:6], 16),
    }


def nearestColor(color):
    r = color["r"]
    g = color["g"]
    b = color["b"]

    nearest = None
    nearest_distance = float("inf")

    for hex_color in intensityColors:
        rgb = hex_to_rgb(hex_color)

        distance = (
            (r - rgb["r"]) ** 2
            + (g - rgb["g"]) ** 2
            + (b - rgb["b"]) ** 2
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest = hex_color

    return nearest


def getIntensity(color):
    c = nearestColor(color)
    index = intensityColors.index(c)
    return math.ceil(((index + 1) / intensityColorsCount) * 100)


def png_to_intensity_grid(image_path):
    """Return a 2D list grid[y][x] of intensities from the PNG at `image_path`.

    Pixels with alpha == 0 map to intensity 0.
    """

    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    pixels = img.load()

    grid = []

    for y in range(height):
        row = []

        for x in range(width):
            r, g, b, alpha = pixels[x, y]

            if alpha > 0:
                intensity = getIntensity({"r": r, "g": g, "b": b})
            else:
                intensity = 0

            row.append(intensity)

        grid.append(row)

    return grid


def png_to_xy_intensity(image_path, include_zero=False):
    """Return a list of {x,y,intensity} records for pixels in the PNG.

    If `include_zero` is False, pixels with intensity 0 are omitted.
    """

    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    pixels = img.load()

    points = []

    for y in range(height):
        for x in range(width):
            r, g, b, alpha = pixels[x, y]

            if alpha > 0:
                intensity = getIntensity({"r": r, "g": g, "b": b})
            else:
                intensity = 0

            if include_zero or intensity > 0:
                points.append({"x": x, "y": y, "intensity": intensity})

    return points


def points_to_intensity_grid(points, width=None, height=None, default=0):
    """Reconstruct a grid[y][x] from {x, y, intensity} point records."""

    if width is None:
        width = max((point["x"] for point in points), default=-1) + 1

    if height is None:
        height = max((point["y"] for point in points), default=-1) + 1

    grid = [
        [default for _ in range(width)]
        for _ in range(height)
    ]

    for point in points:
        x = point["x"]
        y = point["y"]
        intensity = point["intensity"]

        if 0 <= y < height and 0 <= x < width:
            grid[y][x] = intensity

    return grid



def png_to_xy_binary(image_path, include_zero=False):
    """Return a list of {x,y,intensity} records for pixels in the PNG.

    If `include_zero` is False, pixels with intensity 0 are omitted.
    """

    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    pixels = img.load()

    points = []

    for y in range(height):
        for x in range(width):
            r, g, b, alpha = pixels[x, y]

            if alpha > 0:
                #intensity = getIntensity({"r": r, "g": g, "b": b})
                intensity = 1
            else:
                intensity = 0

            if include_zero or intensity > 0:
                points.append({"x": x, "y": y, "intensity": intensity})

    return points