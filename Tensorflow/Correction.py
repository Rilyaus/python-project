import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import pandas as pd
import time

start_time = time.time()

caseNum = '01'
param = 'wind_v'

testSet = open('./Data/Correction/case' + caseNum + '_' + param + '.dat')
trainSet = open('./Data/Correction/case' + caseNum + '_' + param + '.dat')
read_testSet = np.loadtxt(testSet)
read_trainSet = np.loadtxt(trainSet)

testSet.close()
trainSet.close()

#lon_def = 125.079590
#lat_def = 36.578830

repeatCount = 5

batch_size = 10
rnn_size = 10
learing_rate = 0.01

epoch = 700

fileName = 'correction_' + param + '_' + caseNum + '.dat'
ifileName = 'correction_' + param + '_' + caseNum + '_index.dat'


train_result = read_trainSet[:,-1]
train_data = read_trainSet[:,0:10]
test_result = read_testSet[:,-1]
test_data = read_testSet[:,0:10]

train_result = train_result.reshape(len(np.atleast_1d(train_result)), 1)
train_data = train_data.reshape(len(np.atleast_1d(train_data)), 10, 1)

test_result = test_result.reshape(len(np.atleast_1d(test_result)), 1)
test_data = test_data.reshape(len(np.atleast_1d(test_data)), 10, 1)

# ---------- RNN Network ---------- #

# input place holders
data = tf.placeholder(tf.float32, [None, 10, 1])
p_data = tf.placeholder(tf.float32, [None, 1])

# RNN Network
rnn_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
val, _state = tf.nn.dynamic_rnn(rnn_cell, data, dtype=tf.float32)
pred = tf.contrib.layers.fully_connected(val[:, -1], 1, activation_fn=None)

# cost
cost_corr = tf.reduce_mean(tf.square(pred - p_data))

# optimizer
optimizer = tf.train.AdamOptimizer(learing_rate)
train_corr = optimizer.minimize(cost_corr)

# RMSE / MAE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
mae = tf.reduce_mean(tf.abs(targets - predictions))

cIndex = []

def NetworkFunc(step) :
    print "step", step+1, "..."
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

    # Training
        for step in range(epoch) :
            _, step_loss_corr = sess.run([train_corr, cost_corr], feed_dict={data:train_data, p_data:train_result})

            #if step%500 == 0 and step != 0 :
            #    print step,", cost_lon :", step_loss_lon
    # Test
        result = []

        for i in range(0, 10) :
            result.append(test_data[0][i])

        for i in range(len(test_data)) :
            #print(test_data)
            std_val = np.std(test_data[i][0:])
            variation = test_result[i] - test_data[i][-1]

            if std_val < abs(variation) :
                cIndex.append(i+10)
                #print(test_data[i], test_data[i,0])
                test_corr = sess.run(pred, feed_dict={data:[test_data[i]]})
    # Set Predict Value
                #corr_result = test_corr
    # Set Original-Prediction Mid Value
                #corr_result = (test_corr + test_result[i])/2
    # Set ( 2 * Predict + Original ) / 3 Value
                #corr_result = (test_corr * 2 + test_result[i]) / 3
    # Set ( 3 * Predict + Original ) / 4 Value
                corr_result = (test_corr * 3 + test_result[i]) / 4

                result.append(sum(corr_result))
                #print(test_corr, test_result[i], corr_result)
                rest = 10 if (len(test_data)-i) > 10 else (len(test_data)-i)
                num = 1;
                for j in xrange(rest, 0, -1) :
                    if i+num < len(test_data) :
                        test_data[i+num, j-1] = corr_result
                        num += 1
            else :
                result.append(test_result[i])

            #print(i, ':', test_data[i][0:])
            #print(variation)

        #test_corr = sess.run(pred, feed_dict={data:test_data})
        #np.savetxt("file1.csv", test_lon, delimiter=",")
        #mae_lon = sess.run(mae, feed_dict={targets:lon_test, predictions: test_lon})
        #print "MAE_lon :", mae_lon

        #print(type(test_corr))
        #test_corr = 1
    return result

#test_pred = np.array(test_lon, test_lat)

#writer = pd.ExcelWriter(fileName, engine='xlsxwriter')

for count in range(1) :
    test_corr = NetworkFunc(count)
    np.savetxt('./Result/' + fileName, test_corr, delimiter=" ", fmt='%.4f');
    np.savetxt('./Result/' + ifileName, cIndex, delimiter=" ", fmt='%d');
    cIndex = []
    #df = pd.DataFrame(test_corr, columns=['wind_v'])
    #sheet_name = caseNum + '_' + str(count+1)
    #df.to_excel(writer, sheet_name)

#writer.save()

end_time = time.time()

print "Time :", end_time - start_time
