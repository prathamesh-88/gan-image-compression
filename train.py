# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU') 
# if len(gpus) > 0:
#     tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == "__main__":
    from gan import train
    EPOCHS = 10
    train(EPOCHS)

