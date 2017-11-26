from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.framework import tensor_util

# important for this part to specify the port and host
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'Server host:port.')
tf.app.flags.DEFINE_string('model', 'wide_deep',
                           'Model name.')
FLAGS = tf.app.flags.FLAGS

def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model
  request.model_spec.signature_name = 'serving_default'

  # true label value, this is taken from the eval.py portion
  label = 1
  
  # hard coded inputs, however, the hard coded inputs can be easily converted into flexible inputs as well
  # these inputs are derived from observing the signature definition using the saved_model_cli show command
  request.inputs['C1'].CopyFrom(
        tf.contrib.util.make_tensor_proto("68fd1e64", shape=[1]))
  request.inputs['C10'].CopyFrom(
        tf.contrib.util.make_tensor_proto("547c0ffe", shape=[1]))
  request.inputs['C11'].CopyFrom(
        tf.contrib.util.make_tensor_proto("bc8c9f21", shape=[1]))
  request.inputs['C12'].CopyFrom(
        tf.contrib.util.make_tensor_proto("60ab2f07", shape=[1]))
  request.inputs['C13'].CopyFrom(
        tf.contrib.util.make_tensor_proto("46f42a63", shape=[1]))
  request.inputs['C14'].CopyFrom(
        tf.contrib.util.make_tensor_proto("07d13a8f", shape=[1]))
  request.inputs['C15'].CopyFrom(
        tf.contrib.util.make_tensor_proto("18231224", shape=[1]))
  request.inputs['C16'].CopyFrom(
        tf.contrib.util.make_tensor_proto("e6b6bdc7", shape=[1]))
  request.inputs['C17'].CopyFrom(
        tf.contrib.util.make_tensor_proto("e5ba7672", shape=[1]))
  request.inputs['C18'].CopyFrom(
        tf.contrib.util.make_tensor_proto("74ef3502", shape=[1]))
  request.inputs['C19'].CopyFrom(
        tf.contrib.util.make_tensor_proto("0", shape=[1]))
  request.inputs['C2'].CopyFrom(
        tf.contrib.util.make_tensor_proto("2c16a946", shape=[1]))
  request.inputs['C20'].CopyFrom(
        tf.contrib.util.make_tensor_proto("0", shape=[1]))
  request.inputs['C21'].CopyFrom(
        tf.contrib.util.make_tensor_proto("5316a17f", shape=[1]))
  request.inputs['C22'].CopyFrom(
        tf.contrib.util.make_tensor_proto("0", shape=[1]))
  request.inputs['C23'].CopyFrom(
        tf.contrib.util.make_tensor_proto("32c7478e", shape=[1]))
  request.inputs['C24'].CopyFrom(
        tf.contrib.util.make_tensor_proto("9117a34a", shape=[1]))
  request.inputs['C25'].CopyFrom(
        tf.contrib.util.make_tensor_proto("0", shape=[1]))		
  request.inputs['C26'].CopyFrom(
        tf.contrib.util.make_tensor_proto("", shape=[1]))		
  request.inputs['C3'].CopyFrom(
        tf.contrib.util.make_tensor_proto("503b9dbc", shape=[1]))
  request.inputs['C4'].CopyFrom(
        tf.contrib.util.make_tensor_proto("e4dbea90", shape=[1]))
  request.inputs['C5'].CopyFrom(
        tf.contrib.util.make_tensor_proto("f3474129", shape=[1]))
  request.inputs['C6'].CopyFrom(
        tf.contrib.util.make_tensor_proto("13718bbd", shape=[1]))
  request.inputs['C7'].CopyFrom(
        tf.contrib.util.make_tensor_proto("38eb9cf4", shape=[1]))
  request.inputs['C8'].CopyFrom(
        tf.contrib.util.make_tensor_proto("1f89b562", shape=[1]))
  request.inputs['C9'].CopyFrom(
        tf.contrib.util.make_tensor_proto("a73ee510", shape=[1]))	
  request.inputs['I1'].CopyFrom(
        tf.contrib.util.make_tensor_proto(1.0, shape=[1]))	
  request.inputs['I10'].CopyFrom(
        tf.contrib.util.make_tensor_proto(1.0, shape=[1]))
  request.inputs['I11'].CopyFrom(
        tf.contrib.util.make_tensor_proto(1.0, shape=[1]))	
  request.inputs['I12'].CopyFrom(
        tf.contrib.util.make_tensor_proto(0.0, shape=[1]))		
  request.inputs['I13'].CopyFrom(
        tf.contrib.util.make_tensor_proto(0.0, shape=[1]))	
  request.inputs['I2'].CopyFrom(
        tf.contrib.util.make_tensor_proto(4.0, shape=[1]))		
  request.inputs['I3'].CopyFrom(
        tf.contrib.util.make_tensor_proto(2.0, shape=[1]))	
  request.inputs['I4'].CopyFrom(
        tf.contrib.util.make_tensor_proto(0.0, shape=[1]))		
  request.inputs['I5'].CopyFrom(
        tf.contrib.util.make_tensor_proto(0.0, shape=[1]))	
  request.inputs['I6'].CopyFrom(
        tf.contrib.util.make_tensor_proto(0.0, shape=[1]))		
  request.inputs['I7'].CopyFrom(
        tf.contrib.util.make_tensor_proto(1.0, shape=[1]))	
  request.inputs['I8'].CopyFrom(
        tf.contrib.util.make_tensor_proto(0.0, shape=[1]))		
  request.inputs['I9'].CopyFrom(
        tf.contrib.util.make_tensor_proto(0.0, shape=[1]))	
		
  result_future = stub.Predict.future(request, 5.0)
  prediction = result_future.result().outputs

  # Uncomment this if you want to see the output of the entire TensorProto
  # print('Prediction: ' + str(prediction))
  
  # True label value
  print('True label: ' + str(label))
  
  # converting the tensorproto to an Ndarray for extracting output
  probList = tensor_util.MakeNdarray(prediction['probabilities'])[0]
  if probList[0] < probList[1]:
    print("Prediction: 1")
  else:
    print("Prediction: 0")

if __name__ == '__main__':
  tf.app.run()
