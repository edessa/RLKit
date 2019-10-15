
"""
padding is for image size (104, 80)
(obtained by downsample=2, crop_last_row=True)
"""

cnn_specs = dict()

# Too slow, ~150 seconds training per epoch
# Used on 84x84 size images
spec = dict(
    kernel_sizes=[5,3,3,3,3],
    strides=[3,1,1,2,1],
    paddings=[0,1,1,1,1],
    hidden_sizes=[64,64],
    n_channels=[32,64,64,128,128],
)
cnn_specs["0"] = cnn_specs[0] = spec


spec = dict(
    kernel_sizes=[5,3,3],
    strides=[3,1,1],
    paddings=[0,0,0],
    hidden_sizes=[64,64],
    n_channels=[32,64,64],
)
cnn_specs["1"] = cnn_specs[1] = spec
