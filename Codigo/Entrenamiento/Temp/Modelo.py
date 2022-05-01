
#Machine learning
import tensorflow as tf
#from tensorflow.keras import layers, models


def crearModelo2D(p, run):
    print(
        f"Creadno modelo con input ({p['margen'][run]},{p['margen'][run]},{p['canales'][run]})) tipo ({p['outputs']})")
    # Imagen
    input_1 = tf.keras.layers.Input(shape=(p['margen'][run], p['margen'][run], p['canales'][run]))

    # first conv layer :
    conv2d_1 = tf.keras.layers.Conv2D(64, kernel_size=3, activation=tf.keras.activations.relu)(input_1)

    # Second conv layer :
    conv2d_2 = tf.keras.layers.Conv2D(32, kernel_size=3, activation=tf.keras.activations.relu)(conv2d_1)

    # Flatten layer :
    flatten = tf.keras.layers.Flatten()(conv2d_2)

    final = flatten
    listConcat = [flatten]
    listInputs = [input_1]

    if len(p['inputs']) > 2:
        # Agregamos los otros atrbutos
        for attr in p['inputs'][1:]:
            # The other input
            input_x = tf.keras.layers.Input(shape=(1,))
            listConcat.append(input_x)
            listInputs.append(input_x)

        # Concatenate
        final = tf.keras.layers.Concatenate()(listConcat)

    # output
    if p['outputs'] == 'dato':
        output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.softmax)(final)
        dimOutput = 1
    elif p['outputs'] == 'umbral':
        output = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.relu)(final)
        dimOutput = 2
    else:
        print(f"No se pudo crear el modelo outputs no esta bien definido {p['outputs']}")
        return -1

    full_model = tf.keras.Model(inputs=listInputs, outputs=[output])

    print('DONE')

    # print(full_model.summary())
    return full_model


def crearModelo3D(p, run):
    print(
        f"Creando modelo con input ({p['tiempos'][run]},{p['margen'][run]},{p['margen'][run]},{p['canales'][run]})) y ({p['outputs']})...")
    # Imagen
    input_1 = tf.keras.layers.Input(shape=(p['tiempos'][run], p['margen'][run], p['margen'][run], p['canales'][run]))

    # first conv layer :
    conv3d_1 = tf.keras.layers.Conv3D(64, kernel_size=3, activation=tf.keras.activations.relu)(input_1)

    # Second conv layer :
    conv3d_2 = tf.keras.layers.Conv3D(32, kernel_size=3, activation=tf.keras.activations.relu)(conv3d_1)

    # Flatten layer :
    flatten = tf.keras.layers.Flatten()(conv3d_2)

    final = flatten
    listConcat = [flatten]
    listInputs = [input_1]

    if len(p['inputs']) > 2:
        # Agregamos los otros atrbutos
        for attr in p['inputs'][1:]:
            # The other input
            input_x = tf.keras.layers.Input(shape=(1,))
            listConcat.append(input_x)
            listInputs.append(input_x)

        # Concatenate
        final = tf.keras.layers.Concatenate()(listConcat)

    # output
    if p['outputs'] == 'dato':
        output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.softmax)(final)
        dimOutput = 1
    elif p['outputs'] == 'umbral':
        output = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.relu)(final)
        dimOutput = 2
    else:
        print(f"No se pudo crear el modelo outputs no esta bien definido {p['outputs']}")
        return -1

    full_model = tf.keras.Model(inputs=listInputs, outputs=[output])

    print('DONE')
    # print(full_model.summary())
    return full_model

def crearModelo(params,run):
    if params['tiempos'][run] == 1:
        #Se crea un modelo conv2D
        return crearModelo2D(params,run)
    else:
        #Se crea un modelo conv3D
        return crearModelo3D(params,run)

def crearModelTest():
    params = {'inputs' : ['imagen', '99%','altura'],
          'outputs': 'dato',  #umbral o dato
          'lr'     : 0.01,
          'batch'  : 64,
          'epocas' : 1,
          'canales': [3,2,3,1,2,3],
          'tiempos': [6,1,1,6,6,6],
          'margen' : [110,110,110,110,110,110],
          'runs'   : 1
         }

    return crearModelo3D(params,1)