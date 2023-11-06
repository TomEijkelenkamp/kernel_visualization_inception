from matplotlib import pyplot as plt
from tensorflow.keras.applications import InceptionV3

# Load InceptionV3 pre-trained model
model = InceptionV3(weights='imagenet', include_top=False)

# Get the weights of the convolutional layer
filters = model.get_layer('conv2d_1').get_weights()[0]

# Normalize the weights between 0 and 1 for visualization purposes
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first few filters
n_filters, ix = 32, 1
fig = plt.figure(figsize=(20, 20))  # Specify the figure size
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(3):
        # specify subplot and turn off axis
        ax = plt.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1
# Save the figure
fig.savefig('filters_{}.png'.format('conv2d_1'))
plt.close(fig)  # Close the figure to free memory
