# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU') 
# if len(gpus) > 0:
#     tf.config.experimental.set_memory_growth(gpus[0], True)
import os

def check_folder_system():
    os.mkdir("results") if not os.path.isdir("results") else None
    reqd_folders= ['strips', 'image_compare', 'same_img']
    for i in reqd_folders:
        if not os.path.isdir(os.path.join('.', 'results', i)):
            os.mkdir(os.path.join('.', 'results', i))

if __name__ == "__main__":
    check_folder_system()
    from gan import train
    EPOCHS = 10
    train(EPOCHS)

