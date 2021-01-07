## Fully-connected architecture

model = Sequential()
model.add(Dense(256, input_dim = 28*28, activation = relu))
model.add(Dense(256, activation = relu))
model.add(Dense(128, activation = relu))
model.add(Dense(32, activation = relu))
model.add(Dense(10, activation = softmax))

model.compile(loss = categorical_crossentropy, optimizer = rmsprop)
