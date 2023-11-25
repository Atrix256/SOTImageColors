import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import sys

inputFileName = sys.argv[1]
outputFileName = sys.argv[2]
ylimit = float(sys.argv[3])

im = iio.imread(uri=inputFileName)

# tuple to select colors of each channel line
colors = ("red", "green", "blue")

# create the histogram plot, with three lines, one for
# each color
fig = plt.figure()
plt.xlim([0, 256])
for channel_id, color in enumerate(colors):
    histogram, bin_edges = np.histogram(
        im[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=color)

plt.title("Color Histogram")
plt.xlabel("Color value")
plt.ylabel("Pixel count")
plt.ylim(0, ylimit)

fig.savefig(outputFileName, bbox_inches='tight')
