import tensorflow as tf


def get_measures(image_input, image_output):
    img1 = tf.image.decode_image(tf.io.read_file('./images/{}.png'.format(image_input)))
    img2 = tf.image.decode_image(tf.io.read_file('./images/{}.png'.format(image_output)))
    psnr =  tf.image.psnr(img1, img2, max_val=255).numpy()
    img1 = tf.expand_dims(img1, axis=0)
    img2 = tf.expand_dims(img2, axis=0)
    ssim = tf.image.ssim(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03).numpy()[0]
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)
    mse = tf.reduce_mean(tf.square(img1 - img2)).numpy()
    
    return ssim, psnr, mse