import tensorflow as tf


def crearModelo():   
    p = {'inputs'   : ['imagen','99%'],
          'outputs'  : 'umbral',  #umbral o dato
          'lr'       : 0.001,
          'batch'    : 32,        
          'dataset'  : 1,
          'epocas'   : 10,          
          'paciencia': 4,
          'canales'  : [3],
          'tiempos'  : [4],
          'margen'   : [70],
          'runs'     : 1
         }
    run = 0 

    print(f"Creadno modelo con input ({p['margen'][run]},{p['margen'][run]},{p['canales'][run]})) tipo ({p['outputs']})")
    # Imagen
    input_1 = tf.keras.layers.Input(shape=(p['margen'][run],p['margen'][run],p['canales'][run]))
    
    # Convulutional layers
    rescaling = tf.keras.layers.Rescaling(1./65536)(input_1)
    conv2d_1 = tf.keras.layers.Conv2D(128, kernel_size=3,activation=tf.keras.activations.relu)(rescaling)
    mxPool_1 = tf.keras.layers.MaxPooling2D()(conv2d_1)
    dropout_1  = tf.keras.layers.Dropout(0.2)(mxPool_1)
    
    conv2d_2 = tf.keras.layers.Conv2D(64, kernel_size=3,activation=tf.keras.activations.relu)(dropout_1)
    mxPool_2 = tf.keras.layers.MaxPooling2D()(conv2d_2)
    dropout_2  = tf.keras.layers.Dropout(0.1)(mxPool_2)
    
    conv2d_3 = tf.keras.layers.Conv2D(32, kernel_size=3,activation=tf.keras.activations.relu)(dropout_1)
    mxPool_3 = tf.keras.layers.MaxPooling2D()(conv2d_3)
    dropout_3  = tf.keras.layers.Dropout(0.2)(mxPool_3)
    
    conv2d_4 = tf.keras.layers.Conv2D(64, kernel_size=3,activation=tf.keras.activations.relu)(dropout_3)
    mxPool_4 = tf.keras.layers.MaxPooling2D()(conv2d_4)
    dropout_4  = tf.keras.layers.Dropout(0.2)(mxPool_4)
    
    conv2d_5 = tf.keras.layers.Conv2D(32, kernel_size=3,activation=tf.keras.activations.relu)(dropout_4)
    
    
    # Flatten layer :
    flatten = tf.keras.layers.Flatten()(conv2d_5)
    
    final = flatten
    listConcat = [flatten]
    listInputs = [input_1]
    
    if len(p['inputs'])>1:
        #Agregamos los otros atrbutos        
        for attr in p['inputs'][1:]:
            # The other input
            input_x = tf.keras.layers.Input(shape=(1,))
            listConcat.append(input_x)
            listInputs.append(input_x)

            
        # Concatenate
        final = tf.keras.layers.Concatenate()(listConcat)
        
    dense_1 = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu)(final)
    dense_2 = tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu)(dense_1)
    dense_3 = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu)(dense_2)
    
        
    # output
    if p['outputs'] == 'dato':
        output = tf.keras.layers.Dense(units=1)(dense_3)
        dimOutput = 1
    elif p['outputs'] == 'umbral':
        output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(dense_3)
        dimOutput = 2
    else:
        print(f"No se pudo crear el modelo outputs no esta bien definido {p['outputs']}")
        return -1      
    

    full_model = tf.keras.Model(inputs=listInputs, outputs=[output])
    
    print('DONE')
    
    #print(full_model.summary())
    return full_model