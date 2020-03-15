import time 
import tensorflow as tf
import os
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class ConvText(object):
    """ 
    Convolutional multi-label text classification model 
    """
    def __init__(self, seq_length, num_labels, init_embedding_weights):
        
        self.num_labels = num_labels
        self.lr = 0.001
        self.thres = 0.5
        self.dropout = 0.5

        tf.reset_default_graph()
        self.build(seq_length, init_embedding_weights)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        self.val_loss = []
        self.val_f1score = []
        self.val_accuracy = []
        self.val_precision = []
        self.val_recall = []

    def build(self, seq_length, init_embedding_weights):
        '''
        Build tensorflow computation graph
        '''

        # Input placeholders
        self.input_seq = tf.placeholder(tf.int32, (None, seq_length))
        self.target_tags = tf.placeholder(tf.float64, (None, self.num_labels))
        self.pos_weight = tf.placeholder(tf.float64, (None, self.num_labels))
        self.keep_prob = tf.placeholder(tf.float64, shape=())

        # L2 regularization
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.00001)

        # Embedding layer
        E = tf.get_variable('E',
                            initializer=tf.constant(init_embedding_weights),
                            trainable=True)

        # Seq embedding loopkup
        embedded_seq = tf.nn.embedding_lookup(E, self.input_seq)


        # Convolution Layer 1
        convs_1 = []
        filter_sizes = [3,4,5]
        for filter_size in filter_sizes:
            conv_1 = tf.layers.conv1d(inputs=embedded_seq,
                                    filters=32,
                                    kernel_size=filter_size,
                                    strides=1,
                                    activation=tf.nn.relu)
            convs_1.append(conv_1)
        
        merge_1 = tf.concat(convs_1, axis=1)

        # Convolutional Layer 2
        conv_2 = tf.layers.conv1d(inputs=merge_1,
                                filters=32,
                                kernel_size=5,
                                strides=2,
                                activation=tf.nn.relu)

        # # Convolutional Layer 3
        # conv_3 = tf.layers.conv1d(inputs=conv_2,
        #                         filters=16,
        #                         kernel_size=5,
        #                         strides=2,
        #                         activation=tf.nn.relu)
        
        # Dense layer 4
        flatten_4 = tf.layers.flatten(inputs=conv_2)
        dropout_4 = tf.nn.dropout(x=flatten_4,
                                keep_prob=self.keep_prob)
        dense_4 = tf.layers.dense(inputs=dropout_4,
                                units=250,
                                activation=tf.nn.relu,
                                kernel_regularizer=regularizer)
        
        # Dense layer 5
        dropout_5 = tf.nn.dropout(x=dense_4,
                                keep_prob=self.keep_prob)
        logits = tf.layers.dense(dropout_5,
                                units=self.num_labels,
                                kernel_regularizer=regularizer)

        # Pass logits through sigmoid to get probability of each tag
        self.predicted_tags = tf.nn.sigmoid(logits)


        # Pass logits through sigmoid and calculate the cross-entropy
        entropy = tf.nn.weighted_cross_entropy_with_logits(
                                                    targets=self.target_tags, 
                                                    logits=logits,
                                                    pos_weight=self.pos_weight)

        # Get loss and define the optimizer
        self.loss = tf.reduce_mean(entropy) + tf.losses.get_regularization_loss()
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def evaluate(self, x=None, y=None, batch_size=200, save_results=False,
                pos_weight=None):
        '''
        Evaluate a model
        '''
        total_loss, predicted_tags = 0, np.zeros(y.shape)
        
        for i in range(x.shape[0]//batch_size):

            x_batch = x[i*batch_size:(i+1)*batch_size,:]
            y_batch = y[i*batch_size:(i+1)*batch_size,:]

            l, tag_batch = self.sess.run([self.loss, 
                                            self.predicted_tags], 
                                        feed_dict={
                                            self.input_seq:x_batch,
                                            self.target_tags:y_batch,
                                            self.keep_prob:1,
                                            self.pos_weight:pos_weight})

            total_loss += l
            predicted_tags[i*batch_size:(i+1)*batch_size,:] = np.asarray(
                                                        tag_batch>self.thres,
                                                        dtype=np.int32)

        mean_loss = total_loss/(i+1)
        f1, prec, recall, acc = self.evaluate_metrics(y_true=y, 
                                                    y_pred=predicted_tags,
                                                    print_results=False)
        print('Validation evaluation metrics:')
        print('F1_score: {:.4f} - Precision: {:.4f} - Recall: {:.4f} - Acc: {:.4f} - Loss: {:.4f}'.format(
                                            f1, prec, recall, acc, mean_loss))
        
        if save_results:
            self.val_loss.append(mean_loss)
            self.val_f1score.append(f1)
            self.val_precision.append(prec)
            self.val_recall.append(recall)
            self.val_accuracy.append(acc)


    def fit(self, x=None, y=None, batch_size=20, epochs=1, verbose=1, 
            shuffle=True, val_x=None, val_y=None, initial_epoch=0, 
            save_path=None, pos_weight=None):
        '''
        The train function with api similar to Keras
        '''
        
        # Make dir for checkpoints if doesn't exist
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            saver = tf.train.Saver()

        # Initialize to store training parameters
        self.train_epoch = []
        self.train_loss = []
        self.train_f1score = []
        self.train_accuracy = []
        self.train_precision = []
        self.train_recall = []
        
        if np.any(val_x):
            self.val_loss = []
            self.val_f1score = []
            self.val_accuracy = []
            self.val_precision = []
            self.val_recall = []
        
        # Constant pos_weight if not defined
        if pos_weight == None:
            pos_weight = np.ones((1, self.num_labels))

        for e in range(initial_epoch, epochs+initial_epoch):

            if shuffle:
                x, y = sklearn.utils.shuffle(x, y)
            
            print('Epoch {:}/{:}'.format(e+1, epochs+initial_epoch))
            progbar = tf.keras.utils.Progbar(x.shape[0], verbose=verbose)

            total_loss = 0
            predicted_tags = np.zeros(y.shape)
            self.is_training = True
            
            for i in range(x.shape[0]//batch_size):
                
                
                x_batch = x[i*batch_size:(i+1)*batch_size,:]
                y_batch = y[i*batch_size:(i+1)*batch_size,:]

                l, _, tag_batch = self.sess.run([self.loss, 
                                                    self.opt, 
                                                    self.predicted_tags], 
                                                feed_dict={
                                                    self.input_seq:x_batch,
                                                    self.target_tags:y_batch,
                                                    self.keep_prob:self.dropout,
                                                    self.pos_weight:pos_weight})

                progbar.add(x_batch.shape[0], values=[("Loss", l)])
                total_loss += l
                predicted_tags[i*batch_size:(i+1)*batch_size,:] = np.asarray(
                                                        tag_batch>self.thres,
                                                        dtype=np.int32)
            
            print('Training Evaluation Metrics:')
            f1, prec, recall, acc = self.evaluate_metrics(y_true=y, 
                                                        y_pred=predicted_tags,
                                                        print_results=True)
            self.train_epoch.append(e+1)
            self.train_loss.append(total_loss/(i+1))
            self.train_f1score.append(f1)
            self.train_precision.append(prec)
            self.train_recall.append(recall)
            self.train_accuracy.append(acc)

            # Perform Validation
            if np.any(val_x):
                self.evaluate(x=val_x, 
                            y=val_y, 
                            batch_size=batch_size*2, 
                            save_results=True,
                            pos_weight=pos_weight)
            if save_path:
                saver.save(self.sess, save_path,  global_step=e)


    def predict(self, x=None, batch_size=20):
        '''
        Predict tags
        '''
        
        self.is_training = False
        y = np.zeros((x.shape[0], self.num_labels))
        
        for i in range(x.shape[0]//batch_size):

            x_batch = x[i*batch_size:(i+1)*batch_size,:]    
            predicted_tags = self.sess.run([self.predicted_tags], 
                                feed_dict={self.input_seq:x_batch,
                                            self.keep_prob:1})

            y[i*batch_size:(i+1)*batch_size,:] = np.asarray(
                                                predicted_tags[0],
                                                dtype=np.float32)
    
        return y

    
    def evaluate_metrics(self, y_true, y_pred, print_results=True):
        '''
        Metrics to evaluate tag prediction
        '''

        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)

        if print_results:
            print('F1_score: {:.4f} - Precision: {:.4f} - Recall: {:.4f} - Acc: {:.4f}'.format(
                                        f1, precision, recall, accuracy))

        return f1, precision, recall, accuracy
    

    def save(self, save_path='./final_model/my_model.ckpt'):
        
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)
        print("Model saved in path: %s" % save_path)

    
    def restore(self, save_path='./final_model/my_model.ckpt'):
        
        saver = tf.train.Saver()
        saver.restore(self.sess, save_path)
        print("Model restored.")

