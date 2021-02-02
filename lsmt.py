from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam

 
 
def create_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(255, )))
    model.add(Embedding(7799 + 2, 128))
    model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(TimeDistributed(Dense(3)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(0.001),
                metrics=['accuracy'])
    
    model.summary()
    return model
