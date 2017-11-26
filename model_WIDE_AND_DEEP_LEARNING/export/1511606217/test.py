import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.contrib.session_bundle import manifest_pb2
 
def read_serving_signatures(meta_file):
    tf.reset_default_graph()
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(meta_file)
 
        graph = tf.get_default_graph()
        pb = graph.get_collection("serving_signatures")[0]
        signatures = manifest_pb2.Signatures()
        pb.Unpack(signatures)
        
        return signatures
		
print(read_serving_signatures("saved_model.pb"))