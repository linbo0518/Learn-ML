import numpy as np
import matplotlib.pyplot as plt

greyhound_num = 500
lab_num = 500

greyhound_height_data = 28 + 4 * np.random.randn(greyhound_num)
lab_geight_data = 24 + 4 * np.random.randn(lab_num)

plt.hist([greyhound_height_data, lab_geight_data], stacked=True, color=(['r', 'b']))
plt.show()