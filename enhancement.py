import os, time, sys
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf, matplotlib.pyplot as plt


ENHANCEMENT_MODEL_PATH = "models/esrgan"
model = tf.saved_model.load(ENHANCEMENT_MODEL_PATH)


def preprocess_image(image_path: str):
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))

    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)

    # print(hr_size, hr_image)
    return tf.expand_dims(hr_image, 0)

def to_PIL(image) -> Image.Image:
    image = tf.clip_by_value(image, 0, 255)
    return Image.fromarray(tf.cast(image, tf.uint8).numpy())


def save_image(image, filename):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(f"{filename}.jpg")
    print("saved as {}.jpg".format(filename))

def prep_to_plot(image):
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    return image

def plot_image(image, title=""):

    image = prep_to_plot(image)
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()

def multi_plot(images, titles=list()):
    if len(images) > 8:
        return
    for i in range(len(images)):
        plt.subplot(100 + 10 * len(images) + i + 1)
        plt.imshow(prep_to_plot(images[i]))
        plt.axis('off')
        plt.title(titles[i])
    plt.show()

def downscale_image(image):
  """
      Scales down images using bicubic downsampling.
      Args:
          image: 3D or 4D tensor of preprocessed image
  """
  image_size = []
  if len(image.shape) == 3:
    image_size = [image.shape[1], image.shape[0]]
  else:
    raise ValueError("Dimension mismatch. Can work only on single image.")

  image = tf.squeeze(
      tf.cast(
          tf.clip_by_value(image, 0, 255), tf.uint8))

  lr_image = np.asarray(
    Image.fromarray(image.numpy())
    .resize([image_size[0] // 4, image_size[1] // 4],
              Image.BICUBIC))

  lr_image = tf.expand_dims(lr_image, 0)
  lr_image = tf.cast(lr_image, tf.float32)
  return lr_image

def use_model(input_image):
    start = time.time()
    fake_image = model(input_image)
    print("Time Taken: %f" % (time.time() - start))
    return tf.squeeze(fake_image)

def psnr_diff(target, original):
    return tf.image.psnr(
        tf.clip_by_value(target, 0, 255),
        tf.clip_by_value(original, 0, 255),
        max_val= 255
    )


def enhance_image(image: Image.Image) -> Image.Image:
    image = preprocess_image(image)
    image = use_model(image)
    return to_PIL(image)
if __name__ == "__main__":
    IMAGE_PATH = sys.argv[1]
    hr_image = preprocess_image(IMAGE_PATH)

    # plot_image(tf.squeeze(hr_image), "Original Res")
    save_image(tf.squeeze(hr_image), filename="gen/orig")

    lr_image = downscale_image(tf.squeeze(hr_image))
    # plot_image(tf.squeeze(lr_image), title="Low Res")
    save_image(tf.squeeze(lr_image), filename="gen/low_r")

    ge_image = use_model(lr_image)    
    gb = ImageFilter.GaussianBlur()
    # to_PIL(ge_image).filter(filter=ImageFilter.GaussianBlur(1)).save("gen/super_r.jpg")

    # plot_image(tf.squeeze(ge_image), title="Super Res")
    save_image(tf.squeeze(ge_image), filename="gen/super_r")

    print("PSNR Achieved: {}".format(psnr_diff(ge_image, hr_image)[0]))
    multi_plot([tf.squeeze(hr_image),tf.squeeze(lr_image),tf.squeeze(ge_image)], ["Original", "4x nearest", "Super"])


"""
4x Enhancement

"""