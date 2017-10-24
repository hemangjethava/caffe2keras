from __future__ import division, print_function
import caffe
import numpy as np
import os

DATA_DIR = "/home/hemang/deeplearning-cats-dogs-tutorial/caffe_models/"
OUTPUT_DIR = os.path.join(DATA_DIR, "caffe_model_1")

CAFFE_HOME="/home/hemang/caffe/"

MODEL_DIR = os.path.join(CAFFE_HOME, "models", "caffe_model_1")
MODEL_PROTO = os.path.join(MODEL_DIR, "caffenet_deploy_1.prototxt")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "caffe_model_1_iter_500.caffemodel")
MEAN_IMAGE = os.path.join(MODEL_DIR, "mean.binaryproto")

caffe.set_mode_cpu()
net = caffe.Net(MODEL_PROTO, MODEL_WEIGHTS, caffe.TEST)

# layer names and output shapes
for layer_name, blob in net.blobs.iteritems():
    print(layer_name, blob.data.shape)

# write out weight matrices and bias vectors
for k, v in net.params.items():
    print(k, v[0].data.shape, v[1].data.shape)
    np.save(os.path.join(OUTPUT_DIR, "W_{:s}.npy".format(k)), v[0].data)
    np.save(os.path.join(OUTPUT_DIR, "b_{:s}.npy".format(k)), v[1].data)

# write out mean image
blob = caffe.proto.caffe_pb2.BlobProto()
with open(MEAN_IMAGE, 'rb') as fmean:
    mean_data = fmean.read()
blob.ParseFromString(mean_data)
mu = np.array(caffe.io.blobproto_to_array(blob))
print("Mean image:", mu.shape)
np.save(os.path.join(OUTPUT_DIR, "mean_image.npy"), mu)


