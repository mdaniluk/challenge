from readData import Loader
import numpy as np
import time 
import os
import shutil
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN

from sklearn.model_selection import train_test_split


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size)

    for epoch in range(num_epochs):
        print ("Epoch", epoch)
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        
        print (num_batches_per_epoch)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            
class Config():
    embedding_size = 300
    non_static = False
    hidden_size = 300
    max_pool_size = 4
    filter_sizes = "3,4,5"
    num_filters = 128
    l2_reg_lambda = 0.0
    dropout_keep_prob = 1.0
    batch_size = 4
    epochs = 2
    evaluate_every = 10

def train():
    config = Config()
    data_loader = Loader()
    x_raw, y_raw, word_to_id, labels = data_loader.load_data('data/companies.jsons')
    word_embeddings = data_loader.load_embeddings(config.embedding_size)
    id_to_word = data_loader.invert_vocab(word_to_id)
    embedding_mat = [word_embeddings[word] for index, word in enumerate(word_to_id)]
    embedding_mat = np.array(embedding_mat, dtype = np.float32)
    
    x, x_test, y, y_test = train_test_split(x_raw, y_raw, test_size=0.1)
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)
    
    timestamp = str(int(time.time()))
    trained_dir = 'save/trained_results' + timestamp + '/'
    if os.path.exists(trained_dir):
        shutil.rmtree(trained_dir)
    os.makedirs(trained_dir)
    
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn_rnn = TextCNNRNN(
				embedding_mat=embedding_mat,
				sequence_length=x_train.shape[1],
				num_classes = y_train.shape[1],
				non_static=config.non_static,
				hidden_unit=config.hidden_size,
				max_pool_size=config.max_pool_size,
				filter_sizes=map(int, config.filter_sizes.split(",")),
				num_filters = config.num_filters,
				embedding_size = config.embedding_size,
				l2_reg_lambda = config.l2_reg_lambda)
    
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
            grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            
            checkpoint_dir = 'save/checkpoints' + timestamp + '/'
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            
            def real_len(batches):
                return [np.ceil(np.argmin(batch + [0]) * 1.0 /config.max_pool_size) for batch in batches]
                
            def train_step(x_batch, y_batch):
                feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.input_y: y_batch,
					cnn_rnn.dropout_keep_prob: config.dropout_keep_prob,
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, config.embedding_size, 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
                _, step, loss, accuracy = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy], feed_dict)
            
            def dev_step(x_batch, y_batch):
                feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.input_y: y_batch,
					cnn_rnn.dropout_keep_prob: 1.0,
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, config.embedding_size, 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
                step, loss, accuracy, num_correct, predictions = sess.run(
					[global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions], feed_dict)
                return accuracy, loss, num_correct, predictions
            
            saver = tf.train.Saver(tf.all_variables())
            sess.run(tf.initialize_all_variables())
            
            train_batches = batch_iter(list(zip(x_train, y_train)), config.batch_size, config.epochs)
            best_accuracy, best_at_stp = 0, 0
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)
                
                # Evaluate the model with x_dev and y_dev
                if current_step % config.evaluate_every == 0:
                    dev_batches = batch_iter(list(zip(x_dev, y_dev)), config.batch_size, 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        acc, loss, num_dev_correct, predictions = dev_step(x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct
                    accuracy = float(total_dev_correct) / len(y_dev)
                    print (accuracy)
                    if accuracy >= best_accuracy:
                        best_accuracy, best_at_step = accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print ('Best accuracy :  ', best_accuracy )
                    
            saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
            test_batches = batch_iter(list(zip(x_test, y_test)), 1, 1, shuffle=False)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                acc, loss, num_test_correct, predictions = dev_step(x_test_batch, y_test_batch)
                total_test_correct += int(num_test_correct)
            print ('Accuracy on test set ', float(total_test_correct) / len(y_test))
                
                
            
    print ('a')

if __name__ == "__main__":
    train()