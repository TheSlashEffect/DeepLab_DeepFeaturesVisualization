import urllib
import tempfile
from Inference import *


def classifyImages(file_name: str, visualize_middle_layers: bool):
    with open(file_name) as input_file:
        for imageName in input_file.readlines():
            run_visualization(MODEL, 'input/' + imageName.rstrip() + '.jpg',
                              FULL_COLOR_MAP, LABEL_NAMES, visualize_middle_layers)


if __name__ == '__main__':
    LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])

    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
    download_model = False

    if download_model:
        # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug',
        # 'xception_coco_voctrainval']
        # Alternative model you can use. Much less deep, which results in
        # lower (as expected) performance.
        # MODEL_NAME = 'mobilenetv2_coco_voctrainaug'
        MODEL_NAME = 'xception_coco_voctrainval'

        _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
        _MODEL_URLS = {
            'mobilenetv2_coco_voctrainaug':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
            'mobilenetv2_coco_voctrainval':
                'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
            'xception_coco_voctrainaug':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
            'xception_coco_voctrainval':
                'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        }
        _TARBALL_NAME = 'deeplab_model.tar.gz'

        model_dir = tempfile.mkdtemp()
        # tf.gfile.MakeDirs(model_dir) I have no idea what this was originally for.
        # Leaving it be in case one may need it.

        model_tar_path = os.path.join(model_dir, _TARBALL_NAME)
        print('Download path:', model_tar_path)
        print('downloading model, this might take a while...')
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                                   model_tar_path)
        print('download completed! loading DeepLab model...')

    else:
        model_tar_path = 'modelBalls/xception_65.tar.gz'

    MODEL = DeepLabModel(model_tar_path)
    print('model loaded successfully!')

    # Part 1
    classifyImages('inputImagesFull.txt', False)
    # Parts 2 & 3
    classifyImages('inputImagesMiddleLayers.txt', True)
