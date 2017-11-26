from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

CONTINUOUS_COLUMNS =  ["I"+str(i) for i in range(1,14)] # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C"+str(i) for i in range(1,27)] # 1-26 inclusive
LABEL_COLUMN = ["clicked"]

TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

BATCH_SIZE = 400

def generate_input_fn(filename, batch_size=BATCH_SIZE):
    def _input_fn():
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TextLineReader()
        # Reads out batch_size number of lines
        key, value = reader.read_up_to(filename_queue, num_records=batch_size)
        
        # 1 int label, 13 ints, 26 strings
        cont_defaults = [ [0] for i in range(1,14) ]
        cate_defaults = [ [" "] for i in range(1,27) ]
        label_defaults = [ [0] ]
        column_headers = TRAIN_DATA_COLUMNS
        # The label is the first column of the data.
        record_defaults = label_defaults + cont_defaults + cate_defaults

        # Decode CSV data that was just read out. 
        # Note that this does NOT return a dict, 
        # so we will need to zip it up with our headers
        columns = tf.decode_csv(
            value, record_defaults=record_defaults)
        
        # all_columns is a dictionary that maps from column names to tensors of the data.
        all_columns = dict(zip(column_headers, columns))
        
        # Pop and save our labels 
        # dict.pop() returns the popped array of values; exactly what we need!
        labels = all_columns.pop(LABEL_COLUMN[0])
        
        # the remaining columns are our features
        features = all_columns 

        # Sparse categorical features must be represented with an additional dimension. 
        # There is no additional work needed for the Continuous columns; they are the unaltered columns.
        # See docs for tf.SparseTensor for more info
        for feature_name in CATEGORICAL_COLUMNS:
            features[feature_name] = tf.expand_dims(features[feature_name], -1)

        return features, labels

    return _input_fn

print('input function configured')

deep_columns = []
wide_columns = []

# Continuous columns
for name in CONTINUOUS_COLUMNS:
	deep_columns.append(tf.feature_column.numeric_column(name))
	
for name in CATEGORICAL_COLUMNS:
	wide_columns.append(tf.feature_column.categorical_column_with_hash_bucket(name, hash_bucket_size=1000))

# cross columns
crossed_columns = [
    tf.feature_column.crossed_column(
        ['C1', 'C1'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['C4', 'C5', 'C10'], hash_bucket_size=1000),
]	

model_dir = 'models/model_WIDE_AND_DEEP_LEARNING'

runconfig = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))
	  
m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns + crossed_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 70, 50, 25],
            config=runconfig)

from tensorflow.contrib.learn.python.learn import evaluable
isinstance(m, evaluable.Evaluable)

# LOCAL. Update these paths as appropriate
train_file = "data_files/train.csv"
eval_file  = "data_files/eval.csv"

train_sample_size = 800000
train_steps = train_sample_size/BATCH_SIZE # 8000/40 = 200

m.fit(input_fn=generate_input_fn(train_file, BATCH_SIZE), steps=train_steps)

print('fit done')

eval_sample_size = 200000 # this can be found with a 'wc -l eval.csv'
eval_steps = eval_sample_size/BATCH_SIZE # 2000/40 = 50

results = m.evaluate(input_fn=generate_input_fn(eval_file), 
                     steps=eval_steps)
print('evaluate done')

print('Accuracy: %s' % results['accuracy'])
print(results)

def pred_fn():
    sample = [ 0, 127, 1, 3, 1683, 19, 26, 17, 475, 0, 9, 0, 3, "05db9164", "8947f767", "11c9d79e", "52a787c8", "4cf72387", "fbad5c96", "18671b18", "0b153874", "a73ee510", "ceb10289", "77212bd7", "79507c6b", "7203f04e", "07d13a8f", "2c14c412", "49013ffe", "8efede7f", "bd17c3da", "f6a3e43b", "a458ea53", "35cd95c9", "ad3062eb", "c7dc6720", "3fdb382b", "010f6491", "49d68486"]
    sample_dict = dict(zip(FEATURE_COLUMNS, sample))
    
    for feature_name in CATEGORICAL_COLUMNS:
        sample_dict[feature_name] = tf.expand_dims(sample_dict[feature_name], -1)
        
    for feature_name in CONTINUOUS_COLUMNS:
        sample_dict[feature_name] = tf.expand_dims(tf.constant(sample_dict[feature_name], dtype=tf.int32), -1)
    print(sample_dict)

    return sample_dict

print(list(m.predict(input_fn=pred_fn)))

from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

def column_to_dtype(column):
    if column in CATEGORICAL_COLUMNS:
        return tf.string
    else:
        return tf.float32

def serving_input_fn():
    feature_placeholders = {
        column: tf.placeholder(column_to_dtype(column), [None])
        for column in FEATURE_COLUMNS
    }
    # DNNCombinedLinearClassifier expects rank 2 Tensors, but inputs should be
    # rank 1, so that we can provide scalars to the server
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    
    return input_fn_utils.InputFnOps(
        features, # input into graph
        None,
        feature_placeholders # tensor input converted from request 
    )
    
export_folder = m.export_savedmodel("D:\Git Repo\wide-and-deep\models\model_WIDE_AND_DEEP_LEARNING" + '\export',serving_input_fn)

print('model exported successfully to {}'.format(export_folder))
