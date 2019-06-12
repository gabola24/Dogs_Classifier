import tensorflow as tf

def get_lineal_model(num_class):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model

def get_letnet_model(num_class):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=6, kernel_size=(5,5), activation='tanh',
            padding='same', input_shape=(32, 32, 3)
        )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(16, (5,5), activation='tanh'))
    model.add(tf.keras.layers.AveragePooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(120, (5,5), activation='tanh'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(84, activation='tanh'))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model

def get_alex_net(num_class):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding=’valid’))
    model.add(tf.keras.layers.Activation(‘relu’))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding=’valid’))
    model.add(tf.keras.layers.Activation(‘relu’))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=’valid’))
    model.add(tf.keras.layers.Activation(‘relu’))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=’valid’))
    model.add(tf.keras.layers.Activation(‘relu’))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=’valid’))
    model.add(tf.keras.layers.Activation(‘relu’))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, input_shape=(224*224*3,)))
    model.add(tf.keras.layers.Activation(‘relu’))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(4096))
    model.add(tf.keras.layers.Activation(‘relu’))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(num_class))
    model.add(tf.keras.layers.Activation(‘softmax’))
    return(model)
    

