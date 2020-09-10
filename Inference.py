import os
import os.path
import tarfile
import numpy as np
import tensorflow as tf

from Utils import *
from PIL import Image
from matplotlib import gridspec
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    FINAL_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pre-trained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

        with open('nodeNames.txt', 'w') as graph_file:
            for node in self.graph.get_operations():
                graph_file.write('%s\n' % node.name)

    def run(self, image: Image, image_name: str, display_middle_layers: bool):
        return self.run_with_middle_layers(image, image_name) if display_middle_layers \
            else self.run_without_middle_layers(image)

    def run_with_middle_layers(self, image: Image, image_name: str):
        """Runs inference on a single image.

        Args:
          :param image: A PIL.Image object, raw input image.
          :param image_name: Name of the input image file

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        resized_image = resizeImage(image, self.INPUT_SIZE)
        layer_names = ['xception_65/entry_flow/block1/unit_1/xception_module/add:0',  # Entry flow's output
                       # Various stages of the middle flow
                       'xception_65/middle_flow/block1/unit_5/xception_module/add:0',
                       'xception_65/middle_flow/block1/unit_10/xception_module/add:0',
                       'xception_65/middle_flow/block1/unit_16/xception_module/add:0',
                       # Exit flow's first (out of 2) stage'
                       'xception_65/exit_flow/block1/unit_1/xception_module/add:0',
                       'SemanticPredictions:0']  # Final inference results
        display_all_layers = True
        if display_all_layers:
            with open('allLayers.txt') as f:
                layer_names = [line.rstrip() for line in f]
        output_tensors = list(map(lambda tensor_name: self.graph.get_tensor_by_name(tensor_name), layer_names))
        results = self.sess.run(output_tensors, feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})

        deep_feats = results[:-1]
        inference_results = results[-1]
        for i in range(len(deep_feats)):
            deep_feat = deep_feats[i]
            print('Layer name:', layer_names[i])
            print('\tshape:', deep_feat.shape)
            if len(deep_feat.shape) == 4:
                deep_feat = deep_feat[0]

            N = deep_feat.shape[0] * deep_feat.shape[1]
            C = deep_feat.shape[-1]
            X = np.reshape(deep_feat, [N, C])
            num_of_cluster_dims = 8
            X_reduced_rgb = PCA(n_components=3).fit_transform(X)
            X_reduced_k = PCA(n_components=num_of_cluster_dims).fit_transform(X)

            deep_feat_reduced = np.reshape(X_reduced_rgb, [deep_feat.shape[0], deep_feat.shape[1], 3]).astype(np.uint8)
            k_input = np.reshape(X_reduced_k, [deep_feat.shape[0], deep_feat.shape[1], num_of_cluster_dims])
            x, y, z = k_input.shape
            image_2d = k_input.reshape(x * y, z)

            # Since SKLearn 0.23, KMeans uses all cores by default, making n_jobs deprecated
            kmeans_cluster = KMeans(n_clusters=2, random_state=0)
            kmeans_cluster.fit(image_2d)
            _, cluster_labels = kmeans_cluster.cluster_centers_, kmeans_cluster.labels_

            # Since we only have 2 labels, namely 0 and 1, our labels can be used as the pixel values themselves
            segmented_image = np.reshape(cluster_labels, [deep_feat.shape[0], deep_feat.shape[1]])
            segmented_image[segmented_image == 1] = 255

            figure, axisArray = plt.subplots(1, 2)
            figure.suptitle(layer_names[i])
            axisArray[0].imshow(deep_feat_reduced)
            axisArray[1].imshow(segmented_image, cmap='gray')
            axisArray[0].title.set_text('3 dimensions (RGB)')
            axisArray[1].title.set_text('8 dimensions (2 clusters)')
            folder_path = createFolderIfNotExists(image_name)
            plt.savefig('%s/%03d_%s.png' % (folder_path, i, layer_names[i][:-2].replace('/', '_')))
            plt.show()

        write_tensorboard_output = False
        if write_tensorboard_output:
            writer = tf.summary.FileWriter("output", self.sess.graph)
            writer.close()

        seg_map = inference_results[0]
        return resized_image, seg_map

    def run_without_middle_layers(self, image: Image):
        resized_image = resizeImage(image, self.INPUT_SIZE)
        results = self.sess.run(self.FINAL_TENSOR_NAME, feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = results[0]
        return resized_image, seg_map


def run_visualization(MODEL: DeepLabModel, image_path, FULL_COLOUR_MAP, LABEL_NAMES,
                      display_middle_layers: bool):
    """Inferences DeepLab model and visualizes result."""
    try:
        original_im = Image.open(image_path)
    except IOError:
        print('Cannot retrieve image. Please check image path "%s" ' % image_path)
        return

    print('running deeplab on image %s...' % image_path)
    resized_im, seg_map = MODEL.run(original_im, image_path, display_middle_layers)

    vis_segmentation(resized_im, seg_map, FULL_COLOUR_MAP, LABEL_NAMES, image_path)


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image: Image, seg_map, FULL_COLOUR_MAP, LABEL_NAMES, image_name: str):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOUR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    folder_path = createFolderIfNotExists(image_name)
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.savefig('%s/finalResult.png' % folder_path)
    plt.show()
