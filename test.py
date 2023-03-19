import numpy as np


def minibatch_stddev(input_feature_maps):
    # Shape of input feature maps: (N, C, H, W)
    N, C, H, W = input_feature_maps.shape

    # Calculate mean feature maps
    mean_feature_maps = np.mean(input_feature_maps, axis=0, keepdims=True)

  

    # Compute squared differences
    squared_diffs = (input_feature_maps - mean_feature_maps) ** 2

    # Calculate mean squared differences
    mean_squared_diffs = np.mean(squared_diffs, axis=0, keepdims=True)

    # Compute standard deviation feature maps
    stddev_feature_maps = np.sqrt(mean_squared_diffs)
    print(stddev_feature_maps.shape)
    # Average standard deviation over all channels and spatial locations
    avg_stddev = np.mean(stddev_feature_maps)
    print(avg_stddev)
    print(avg_stddev.shape)
    # Create new feature map with the same spatial dimensions
    new_feature_map = np.full((N, 1, H, W), avg_stddev)

    # Concatenate new feature map with the original feature maps
    extended_feature_maps = np.concatenate((input_feature_maps, new_feature_map), axis=1)

    return extended_feature_maps
fm = np.random.randint(low=1,high=10,size=(200,3,128,64))

minibatch_stddev(fm)

