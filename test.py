import tensorflow as tf

# Check available physical GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

# Additional information about each GPU
for gpu in gpus:
    print("GPU Name:", gpu.name)
    print("GPU Type:", gpu.device_type)
