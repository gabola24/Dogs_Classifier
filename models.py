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
    model.add(tf.keras.layers.Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding= 'valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, input_shape=(224*224*3,)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(4096))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(num_class))
    model.add(tf.keras.layers.Activation('softmax'))
    return(model)
    
def getVGG(num_class):
    model = tf.keras.Sequential()
    # capa 1 good
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(224,224,3),strides=1,padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 2
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 3
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 4
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 5
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #flatten
    model.add(tf.keras.layers.Flatten())
    #dense 1 
    model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
    #dense 2
    model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
    #dense 3 
    model.add(tf.keras.layers.Dense(units=1000,activation="relu"))
    #capa final
    model.add(tf.keras.layers.Dense(units=num_class,activation="softmax"))
    return model

def get_vgg16_model(num_class):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3,3), activation='relu',
            padding='same', input_shape=(224, 224, 3)
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model

def get_google_net_model(num_class):

    input_layer = tf.keras.layers.Input(shape=(224, 224, 3))

    x = tf.keras.layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2')(input_layer)
    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = tf.keras.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)
    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    x = inception_module(x, 192, 96, 208, 16, 48, 64)

    x1 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x1 = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.7)(x1)
    x1 = tf.keras.layers.Dense(num_class, activation='softmax', name='auxilliary_output_1')(x1)

    x = inception_module(x, 160, 112, 224, 24, 64, 64)
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    x = inception_module(x, 112, 144, 288, 32, 64, 64)

    x2 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x2 = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Dense(1024, activation='relu')(x2)
    x2 = tf.keras.layers.Dropout(0.7)(x2)
    x2 = tf.keras.layers.Dense(num_class, activation='softmax', name='auxilliary_output_2')(x2)

    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = inception_module(x, 384, 192, 384, 48, 128, 128)

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(num_class, activation='softmax', name='output')(x)

    return tf.keras.Model(input_layer, [x, x1, x2], name='inception_v1')
    #return tf.keras.Model(input_layer, x, name='inception_v1')
    

def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):

    conv1 = tf.keras.layers.Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)

    conv3 = tf.keras.layers.Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
    conv3 = tf.keras.layers.Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)

    conv5 = tf.keras.layers.Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
    conv5 = tf.keras.layers.Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)

    pool = tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    pool = tf.keras.layers.Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)

    layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out
def get_vgg16_model_experiment(num_class,arch):
    model = tf.keras.Sequential()
    for i in arch:
        model.add(tf.keras.layers.Conv2D(
            filters=arch[i], kernel_size=(3,3), activation='relu',
            padding='same', input_shape=(224, 224, 3)))
            
        model.add(tf.keras.layers.Conv2D(
            filters=arch[i], kernel_size=(3,3), activation='relu',
            padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
            
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048, activation='relu'))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model

def build_standard_cnn(
    num_filters_per_convolutional_layer,
    num_units_per_dense_layer,
    input_shape,
    num_classes,
    activation='relu',
    maxpool=None):
    """
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=num_filters_per_convolutional_layer[0],
            kernel_size=(3, 3), activation=activation,
            padding='same', input_shape=input_shape)
        )
    if maxpool != None:
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    for num_filters in num_filters_per_convolutional_layer[1:]:
        if num_filters==0:
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        else:
            model.add(
                tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=(3, 3), activation=activation,
                    padding='same')
            )
        
    model.add(tf.keras.layers.Flatten())
    for num_units in num_units_per_dense_layer:
        if num_filters==0:
            model.add(tf.keras.layers.Dropout(0.4))
        else:
            model.add(tf.keras.layers.Dense(num_units, activation=activation))
        
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    return model