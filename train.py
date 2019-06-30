import numpy as np
import tensorflow as tf
from DATA_PREP import prepare_data
from test_data_prep import prepare_test_data

train_file_directory = r'C:\Users\MSabry\Desktop\cats and dogs classifier\dataset\training_set'
test_file_directory = r'C:\Users\MSabry\Desktop\cats and dogs classifier\dataset\test_set'
x_train,x_cv,y_train,y_cv = prepare_data(train_file_directory)
x_test,y_test = prepare_test_data(test_file_directory)
        

tf.reset_default_graph()
filter_1_size = 3
filter_2_size = 3
filter_3_size = 3
filter_4_size = 3
filter_5_size = 3
raw_img_depth = 3
image_size = 128
hl1_filters = 32
hl2_filters = 64
hl3_filters = 96
hl4_filters = 128
hl5_filters = 128
fc1_units = 1024
fc2_units = 128
out_units = 1
epochs = 100
batch_size = 128
test_batch_size = 128
conv_layers = 3
beta = 0.01

#generating weights:
initializer = tf.contrib.layers.xavier_initializer_conv2d()
weights = {'filter_1':tf.Variable(initializer(shape = [filter_1_size,filter_1_size,raw_img_depth,hl1_filters]),
                                  name = 'filter_1'),
           'filter_2':tf.Variable(initializer(shape = [filter_2_size,filter_2_size,hl1_filters,hl2_filters]),
                                  name = 'filter_2'),
           'filter_3':tf.Variable(initializer(shape = [filter_3_size,filter_3_size,hl2_filters,hl3_filters]),
                                  name = 'filter_3'),
                                  
           'filter_4':tf.Variable(initializer(shape = [filter_4_size,filter_4_size,hl3_filters,hl4_filters]),
                                  name = 'filter_4'),
                                  
           'filter_5':tf.Variable(initializer(shape = [filter_5_size,filter_5_size,hl4_filters,hl5_filters]),
                                  name = 'filter_5'),
                                   
           'W_fc1':tf.Variable(initializer(shape = [16*16*hl5_filters,fc1_units]),
                                  name = 'W_fc1'),
           'W_fc2':tf.Variable(initializer(shape = [fc1_units,fc2_units]),
                                  name = 'W_fc1'),
           'W_out':tf.Variable(initializer(shape = [fc2_units,out_units]),
                                  name = 'W_out')}

biases = {'bias_conv_1':tf.Variable(tf.zeros(shape = [hl1_filters]),name = 'bias_conv_1'),
          'bias_conv_2':tf.Variable(tf.zeros(shape = [hl2_filters]),name = 'bias_conv_2'),
          'bias_conv_3':tf.Variable(tf.zeros(shape = [hl3_filters]),name = 'bias_conv_3'),
          'bias_conv_4':tf.Variable(tf.zeros(shape = [hl4_filters]),name = 'bias_conv_4'),
          'bias_conv_5':tf.Variable(tf.zeros(shape = [hl5_filters]),name = 'bias_conv_5'),
          'bias_fc_1':tf.Variable(tf.zeros(shape = [fc1_units]),name = 'bias_fc_1'),
          'bias_fc_2':tf.Variable(tf.zeros(shape = [fc2_units]),name = 'bias_fc_2'),
          'bias_out':tf.Variable(tf.zeros(shape = [out_units]),name = 'bias_out')}



def nn_model(data,weights,biases,training = True):
    training = training
    with tf.name_scope('layer_1') as scope:
        conv1 = tf.nn.conv2d(data,weights['filter_1'],strides = [1,1,1,1],padding = 'SAME',name = 'conv1')
        bias_added_1 = tf.nn.bias_add(conv1,biases['bias_conv_1'],name = 'bias_added_1')
        relued1 = tf.nn.relu(bias_added_1,name = 'relued1')
        max_pooled_1 = tf.nn.max_pool(relued1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID',name = 'max_pooled_1') #64*64*4

    with tf.name_scope('layer_2') as scope:
        conv2 = tf.nn.conv2d(max_pooled_1,weights['filter_2'],strides = [1,1,1,1],padding = 'SAME',name = 'conv2')
        bias_added_2 = tf.nn.bias_add(conv2,biases['bias_conv_2'],name = 'bias_added_2')
        relued2 = tf.nn.relu(bias_added_2,name = 'relued2')
        max_pooled_2 = tf.nn.max_pool(relued2,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID',name = 'max_pooled_2') #64*64*8
        
    with tf.name_scope('layer_3') as scope:
        conv3 = tf.nn.conv2d(max_pooled_2,weights['filter_3'],strides = [1,1,1,1],padding = 'SAME',name = 'conv3')
        bias_added_3 = tf.nn.bias_add(conv3,biases['bias_conv_3'],name = 'bias_added_3')
        relued3 = tf.nn.relu(bias_added_3,name = 'relued3')
        max_pooled_3 = tf.nn.max_pool(relued3,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID',name = 'max_pooled_3') #64*64*8
        
    with tf.name_scope('layer_4') as scope:
        conv4 = tf.nn.conv2d(max_pooled_3,weights['filter_4'],strides = [1,1,1,1],padding = 'SAME',name = 'conv4')
        bias_added_4 = tf.nn.bias_add(conv4,biases['bias_conv_4'],name = 'bias_added_4')
        relued4 = tf.nn.relu(bias_added_4,name = 'relued4')
        
    with tf.name_scope('layer_5') as scope:
        conv5 = tf.nn.conv2d(relued4,weights['filter_5'],strides = [1,1,1,1],padding = 'SAME',name = 'conv5')
        bias_added_5 = tf.nn.bias_add(conv5,biases['bias_conv_5'],name = 'bias_added_5')
        relued5 = tf.nn.relu(bias_added_5,name = 'relued5')
        
    
    with tf.name_scope('fc_1') as scope:
        reshaped_mx_pl_3 = tf.reshape(relued5,[-1,16*16*hl5_filters])
        fc_1 = tf.add(tf.matmul(reshaped_mx_pl_3,weights['W_fc1']),biases['bias_fc_1'],name = 'fc_1')
        relu_fc_1 = tf.nn.relu(fc_1,name = scope)
        relu_fc_1 = tf.contrib.layers.batch_norm(relu_fc_1)
        if training:
            relu_fc_1 = tf.nn.dropout(relu_fc_1,0.5)
        
        
    with tf.name_scope('fc_2') as scope:
        fc_2 = tf.add(tf.matmul(relu_fc_1,weights['W_fc2']),biases['bias_fc_2'],name = 'fc_2')
        relu_fc_2 = tf.nn.relu(fc_2,name = scope)
        relu_fc_2 = tf.contrib.layers.batch_norm(relu_fc_2)
        if training:
            relu_fc_2 = tf.nn.dropout(relu_fc_2,0.7)
    
        
    with tf.name_scope('output') as scope:
        output = tf.add(tf.matmul(relu_fc_2,weights['W_out']),biases['bias_out'],name = scope)
        
        
    return output
        



regs = tf.nn.l2_loss(weights['W_fc1'])+tf.nn.l2_loss(weights['W_fc2'])+tf.nn.l2_loss(weights['W_out'])
    
X = tf.placeholder( dtype = tf.float32, name = 'X',shape = [None,128,128,3])
Y = tf.placeholder(dtype = tf.float32, name = 'Y',shape =[None,1])


#model prediction
predictions = nn_model(X,weights,biases,training = False)

#evaluation metric
correct = tf.equal(tf.round(tf.nn.sigmoid(predictions)),Y)
acc = tf.reduce_mean(tf.cast(correct,dtype = 'float32'))

#learning rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.001,global_step = global_step,decay_steps = 1000, decay_rate = 0.95)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = predictions ,labels = Y ))
cost = cost + beta * regs

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = global_step)

lr = []
train_accuracy = []
cv_accuracy = []
test_accuracy = []
cost_per_epoch = []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        train_correct_list = []
        epoch_loss = 0
        for i in range(len(x_train)//batch_size):
            print('Batch', i+1, 'Started out of ', len(x_train)//batch_size, 'Batches')
            start = i * batch_size
            end = start+batch_size
            batch_x = x_train[start:end]
            batch_y = y_train[start:end]
            
            _,loss,train_correct = sess.run([optimizer,cost,correct],feed_dict = {'X:0':batch_x,'Y:0':batch_y})
            train_correct_list.append(train_correct)
            
            
            batch_loss = loss
            epoch_loss += loss
             
        train_correct_list = np.where(np.ravel(train_correct_list) == True,1.,0.)
        
        print('epoch: ',epoch,'finished out of:',epochs)
        print('train_acc: ',sess.run(tf.reduce_mean(tf.constant(train_correct_list))))
        print('cv_acc: ',acc.eval(feed_dict = {'X:0':x_cv,'Y:0':y_cv}))
    
    
        test_correct_list = []
        for i in range(len(x_test)//test_batch_size):
            start = i * test_batch_size
            end = start + test_batch_size
            
            batch_x_test = x_test[start:end]
            batch_y_test = y_test[start:end]
            
            test_correct_ = sess.run([correct],feed_dict = {'X:0':batch_x_test,'Y:0':batch_y_test})
            test_correct_list.append(test_correct_)
        
        test_correct_list = np.where(np.ravel(test_correct_list) == True,1.,0.)
        test_acc = np.sum(test_correct_list)/len(test_correct_list)
        print('test_acc: ',test_acc)
        print('loss: ',epoch_loss)
        print('lr: ',sess.run(learning_rate))
        
        
        lr.append(sess.run(learning_rate))
        train_accuracy.append(sess.run(tf.reduce_mean(tf.constant(train_correct_list))))
        cv_accuracy.append(acc.eval(feed_dict = {'X:0':x_cv,'Y:0':y_cv}))
        test_accuracy.append(test_acc)
        cost_per_epoch.append(epoch_loss)


        
        
  
 

