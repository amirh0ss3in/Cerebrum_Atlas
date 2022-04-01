import numpy as np
from CA.utils import plot_coordinates

MNI_coords = np.array([2, 4, 60])
plot_coordinates(MNI_coords, input_coordinate = 'mni', colored = True)
