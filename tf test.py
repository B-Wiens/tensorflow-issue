import tensorflow as tf
import keras
from keras.layers import Dense, Input, Reshape, Concatenate
from keras.models import Model
import numpy as np

# required parameters
num_sources = 2
input_num = 5
output_num = 4

# create list of model inputs
sources = []
for i in np.arange(num_sources) :
    sources.append( Input(shape=(input_num,)) )

# create list to hold model results
results = []

# build model
for i in np.arange(num_sources) :
    r1 = Dense(output_num)(sources[i])
    r2 = Reshape( (1, output_num) )(r1)
    results.append(r2)

result_complete = Concatenate(axis=1)(results)

# compile model
model = Model(inputs=[i for i in sources], outputs=[result_complete])
model.compile(loss='mse', optimizer='adam')
model.summary()

# batch size and number of batchs
batch_size = 5
num_batches = 10

# generate arbitrary data
x = []
y = []
x_tmp = np.array( [ np.arange(input_num) for i in np.arange(num_sources) ] )
y_tmp = np.array( [ np.arange(output_num) for i in np.arange(num_sources) ] )
for i in np.arange(batch_size*num_batches) :
    x.append( x_tmp+i )
    y.append( y_tmp+i )
#print(np.shape(x))
x = list( np.transpose(x, axes=(1,0,2)) )
#y = np.transpose(y, axes=(1,0,2))
y = np.array(y)

model.fit(x, y, batch_size=batch_size, epochs=5)



