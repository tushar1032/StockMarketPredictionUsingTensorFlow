import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

name = ""
learning_rate = 0
num_inputs = 1
num_time_steps = 7
num_neurons = 100
num_outputs = 1
num_train_iterations = 10000
batch_size = 1

x = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])
scaler = MinMaxScaler()

stock_data = pd.DataFrame()
train_data = pd.DataFrame()
test_data = pd.DataFrame()
train_scaled = pd.DataFrame()
test_scaled  = pd.DataFrame()
test_size = 7

dict = {'GOOG':0.00006,'AAPL':0.00008,'AMZN':0.009}


cell = tf.contrib.rnn.OutputProjectionWrapper(
tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()



def setName(nam):
   global name
   name = nam

def Preparation():
    global stock_data
    global train_size
    global train_data
    global test_data
    global train_scaled
    global test_scaled
    global size_of_csv
    global learning_rate

    stock_data = pd.read_csv( name+'1year.csv',index_col='Date')
    stock_data.index = pd.to_datetime(stock_data.index)

    stock_data = stock_data.drop(['Open','Low','Close','Volume','Adj Close'],axis = 1)

    size_of_csv = stock_data.shape[0]
    train_size = size_of_csv - test_size

    train_data = stock_data.head(train_size)
    test_data = stock_data.tail(test_size)

    learning_rate = dict[name]

    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)



def next_batch(training_data,batch_size,steps):
    rand_start = np.random.randint(0,len(training_data) - steps)
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


def Model_Training():
    with tf.Session() as sess:
        sess.run(init)

        for iteration in range(num_train_iterations):

            x_batch, y_batch = next_batch(train_scaled,batch_size,num_time_steps)
            sess.run(train, feed_dict={x: x_batch, y: y_batch})

            if iteration % 100 == 0:

                mse = loss.eval(feed_dict={x: x_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)

        # Save Model for Later
        saver.save(sess, "./Project_Model_RNN_Saved" + name)



def Predict():
    with tf.Session() as sess:

        # Use your Saver instance to restore your saved rnn time series model
        saver.restore(sess, "./Project_Model_RNN_Saved"+name)

        # Create a numpy array for your genreative seed from the last 12 months of the
        # training set data. Hint: Just use tail(12) and then pass it to an np.array
        train_seed = list(train_scaled[-test_size:])

        ## Now create a for loop that
        for iteration in range(test_size):
            x_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
            y_pred = sess.run(outputs, feed_dict={x: x_batch})
            train_seed.append(y_pred[0, -1, 0])

    results = scaler.inverse_transform(np.array(train_seed[test_size:]).reshape(test_size,1))

    test_data['Predicted_High'] = results

    print(test_data)

    test_data.plot()
    plt.show()
   # plt.savefig(name + '.png')

dict1 = {1:'GOOG',2:'AAPL',3:'AMZN'}
while True:
    print("1.GOOG\n2.AAPL\n3.AMZN\n4.Exit")
    ch = input("Enter Choice\n\n")
    if int(ch) == 4:
        break
    setName(dict1[int(ch)])
    Preparation()
   # Model_Training()
    Predict()


print("Exited............")



