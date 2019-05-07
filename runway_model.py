import numpy as np
import runway
from utils import predict, load_images, display_images
from loss import depth_loss_function
from layers import BilinearUpSampling2D
from keras.models import load_model
import os
import glob
import argparse
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'


@runway.setup(options={'model_file': runway.file(extension='.h5')})
def setup(opts):
    print('Loading model...')
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D,
                      'depth_loss_function': depth_loss_function}
    graph = tf.get_default_graph()
    model = load_model(
        opts['model_file'],
        custom_objects=custom_objects,
        compile=False
    )
    print('Model loaded')
    return graph, model


@runway.command('predict_depth', inputs={'image': runway.image}, outputs={'depth_image': runway.image(channels=1)})
def predict_depth(graph_and_model, inputs):
    graph, model = graph_and_model
    img = inputs['image']
    original_size = img.size
    img = np.clip(np.asarray(img.resize((640, 480)), dtype=float) / 255, 0, 1)
    img = np.expand_dims(img, 0)
    with graph.as_default():
        outputs = predict(model, img)
    return Image.fromarray(np.uint8(np.squeeze(outputs) * 255), 'L').resize(original_size)


if __name__ == '__main__':
    runway.run(debug=True, model_options={'model_file': 'nyu.h5'})
