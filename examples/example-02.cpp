## CNN architecture 

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation = relu))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation = relu))
model.add(Conv2D(32, (3, 3), activation = relu))
model.add(Flatten())
model.add(Dense(32, activation = relu))
model.add(Dense(10, activation = softmax))

model.compile(loss = categorical_crossentropy, optimizer = rmsprop)
