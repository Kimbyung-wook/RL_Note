import tensorflow as tf

def gpu_memory_limiter(mb:int = 1024):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mb)])
    except RuntimeError as e:
      print(e)