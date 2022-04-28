import tensorflow as tf


def get_measures(image_input, image_output):
    img1 = tf.image.decode_image(tf.io.read_file('./images/{}.png'.format(image_input)))
    img2 = tf.image.decode_image(tf.io.read_file('./images/{}.png'.format(image_output)))
    s1, s2 = sum(img1.shape), sum(img2.shape)
    try:
        if s1 > s2:
            psnr =  tf.image.psnr(img1[img1.shape[0] - img2.shape[0]:, img2.shape[1] - img2.shape[1]:, :], img2, max_val=255).numpy()
            img1 = tf.expand_dims(img1[img1.shape[0] - img2.shape[0]:, img2.shape[1] - img2.shape[1]:, :], axis=0)

        else:
            psnr =  tf.image.psnr(img1, img2[img2.shape[0] - img1.shape[0]:, img2.shape[1] - img1.shape[1]:, :], max_val=255).numpy()
            img2 = tf.expand_dims(img2[:img1.shape[0], :img1.shape[1], :], axis=0)
    except:
        psnr = -1
    # psnr =  tf.image.psnr(img1, img2, max_val=255).numpy()
    try:
        ssim = tf.image.ssim(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03).numpy()[0]
    except:
        ssim = -1
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)
    try:
        mse = tf.reduce_mean(tf.square(img1 - img2)).numpy()
    except:
        mse = -1
    
    return ssim, psnr, mse