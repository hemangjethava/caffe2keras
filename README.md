# caffe2keras
conversion of caffemodel to keras weight files.

conversion scripts were referred from sujit pal's blogpost,
http://sujitpal.blogspot.in/2017/01/migrating-vgg-cnn-from-caffe-to-keras.html

caffeconv will convert caffemodel's weights and bias values to weights.npy and bias.npy files for each layer of the network.
convkeras will take these npy files as an input to create h5 file for keras.
