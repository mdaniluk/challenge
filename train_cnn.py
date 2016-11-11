from readData import Loader
import numpy as np
import time 
import os
import shutil
import tensorflow as tf
from text_cnn import TextCNN
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
    embedding_size = 128
    non_static = False
    hidden_size = 128
    max_pool_size = 4
    filter_sizes = "3,4,5"
    num_filters = 128
    l2_reg_lambda = 0.0
    dropout_keep_prob = 1.0
    batch_size = 96
    epochs = 2
    evaluate_every = 5

            
def train_cnn():
    config = Config()
    data_loader = Loader()
    x_raw, y_raw, word_to_id, labels = data_loader.load_data('data/companies.jsons')
    x, x_test, y, y_test = train_test_split(x_raw, y_raw, test_size=0.1)
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
				sequence_length=x_train.shape[1],
				num_classes=y_train.shape[1],
				vocab_size=len(word_to_id),
				embedding_size=config.embedding_size,
				filter_sizes=list(map(int, config.filter_sizes.split(","))),
				num_filters=config.num_filters,
				l2_reg_lambda=config.l2_reg_lambda)
        
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "save/trained_model_" + timestamp))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "save/checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())
            
            def train_step(x_batch, y_batch):
                feed_dict = {
					cnn.input_x: x_batch,
					cnn.input_y: y_batch,
					cnn.dropout_keep_prob: config.dropout_keep_prob}
                _, step, loss, acc, num_correct = sess.run([train_op, global_step, cnn.loss, cnn.accuracy, cnn.num_correct], feed_dict)
                return num_correct
            
            def dev_step(x_batch, y_batch):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
                step, loss, acc, num_correct = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct], feed_dict)
                return num_correct
                
            sess.run(tf.initialize_all_variables())
            train_batches = batch_iter(list(zip(x_train, y_train)), config.batch_size, config.epochs)
            best_accuracy, best_at_step = 0, 0
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                num_correct = train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)
                
                accuracy_train = float(num_correct) / len(y_train_batch)
                print ("train accuracy " , accuracy_train)
                
                if current_step % config.evaluate_every == 0:
                    dev_batches = batch_iter(list(zip(x_dev, y_dev)), config.batch_size, 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        num_dev_correct = dev_step(x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct
                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    print ("Validation acc " ,dev_accuracy)
                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print ('Best accuracy :  ', best_accuracy )
                        
                            
            test_batches = batch_iter(list(zip(x_test, y_test)), 1, 1, shuffle=False)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                acc, loss, num_test_correct, predictions = dev_step(x_test_batch, y_test_batch)
                total_test_correct += int(num_test_correct)
            print ('Accuracy on test set ', float(total_test_correct) / len(y_test))
                



    

if __name__ == "__main__":
    train_cnn()