import os
import ntpath
from PIL import Image


def resizeImage(image: Image, input_size: int):
    width, height = image.size
    resize_ratio = 1.0 * input_size / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    return resized_image


def createFolderIfNotExists(image_name):
    """
    Creates directory named after the image's filename (strips extension),
    and returns said name
    :param image_name: The path of the input image
    :return: destination folder's path
    """
    folder_path = os.path.splitext(ntpath.basename(image_name))[0]
    folder_path = 'imageOutput/' + folder_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path
