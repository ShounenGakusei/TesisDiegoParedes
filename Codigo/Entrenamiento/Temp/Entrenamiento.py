import tensorflow as tf
import wandb
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback

# Transformamos un filename tensor en una imagen
from Entrenamiento.Modelo import crearModelo


def read_png_file(item, value, p, run, path_base, products, times):
    # imagenData[0] = XO     # imagenData[1] = XA     # imagenData[2] = Fecha
    imagenData = tf.strings.split(item['imagen'], sep='--')
    size = int(p['margen'][run] / 2)

    timeJoin = []
    for j in range(p['tiempos'][run] - 1, -1, -1):
        filename = path_base + 'PNG/' + imagenData[2] + '/' + imagenData[2] + '_' + str(j) + '.png'

        image_string = tf.io.read_file(filename)

        img_decoded = tf.io.decode_png(image_string, dtype=tf.uint16, channels=3)
        # print(img_decoded.shape)

        timeJoin.insert(0, img_decoded[int(imagenData[1]) - size:int(imagenData[1]) + size,
                           int(imagenData[0]) - size:int(imagenData[0]) + size,
                           0:p['canales'][run]])
    # return timeJoin

    if p['tiempos'][run] == 1:
        imagenData = tf.reshape(timeJoin[0], (p['margen'][run], p['margen'][run], p['canales'][run]))
    else:
        img = tf.stack(timeJoin, axis=0)
        imagenData = tf.reshape(img, (p['tiempos'][run], p['margen'][run], p['margen'][run], p['canales'][run]))

    if len(p['inputs']) == 1:
        return imagenData, int(value)

    item['imagen'] = imagenData
    itemL = []
    for inpL in p['inputs']:
        itemL.append(item[inpL])

    return tuple(itemL), int(value)


def splitDataset(p, run, dataset, path_imagenes, products, times, val_split=0.2):
    # Dataset de etnrenamiento
    train, test = train_test_split(dataset, test_size=val_split, shuffle=True)
    print(f'Tamaño del dataset: Train {len(train)}  - Val {len(test)}')

    inputsList = {}
    for inp in p['inputs']:
        inputsList[inp] = train[inp].tolist()

    train_dataset = tf.data.Dataset.from_tensor_slices(((inputsList), train[p['outputs']].tolist()))
    val_dataset = tf.data.Dataset.from_tensor_slices(((inputsList), train[p['outputs']].tolist()))

    train_dataset = train_dataset.map(lambda x, y: read_png_file(x, y, p, run, path_imagenes, products, times))
    val_dataset = val_dataset.map(lambda x, y: read_png_file(x, y, p, run, path_imagenes, products, times))

    train_dataset = train_dataset.batch(p['batch']).cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(p['batch']).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


def getMetrics(modelType, lr):
    if modelType == 'umbral':
        optimizer = keras.optimizers.RMSprop(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        train_acc_metric = keras.metrics.SparseCategoricalCrossentropy()
        val_acc_metric = keras.metrics.SparseCategoricalCrossentropy()
        metrics = ['acc']

        return optimizer, loss_fn, train_acc_metric, val_acc_metric, metrics
    if modelType == 'dato':
        optimizer = keras.optimizers.RMSprop(learning_rate=1e-3)
        loss_fn = keras.losses.MeanAbsoluteError()
        train_acc_metric = keras.metrics.MeanAbsoluteError()
        val_acc_metric = keras.metrics.MeanAbsoluteError()
        metrics = ['mae']

        return optimizer, loss_fn, train_acc_metric, val_acc_metric, metrics

    return -1, -1, -1, -1, -1


def trainModel(params, dataset, path_base, products, times, val_split=0.2):
    config = dict(learning_rate=params['lr'], epochs=params['epocas'],
                  batch_size=params['batch'], architecture="CNN", )

    resultados = []
    for run in range(params['runs']):
        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
        wandb.init(project='Tesis-DiegoJN', config=config, name=f"Experimetno_{run}")

        # Metricas y parametros de entrenaiento
        optimizer, loss_fn, train_acc_metric, val_acc_metric, metrics = getMetrics(params['outputs'], params['lr'])
        logs = Callback()

        # Modelo
        model = crearModelo(params, run)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics, )

        # Dataset
        train_dataset, val_dataset = splitDataset(params, run, dataset, path_base, products, times, val_split)

        print(f'Inicio de la prueba N°: {run}/{params["runs"]}')
        print(f'- Cantidad de dataset: Train = {len(train_dataset)} - Val = {len(val_dataset)} ')
        print(f'- Numero batch:  {params["batch"]}')

        # Entrenamos
        history = model.fit(train_dataset, batch_size=params['batch'],
                            epochs=params['epocas'], callbacks=[logs],
                            validation_data=val_dataset,
                            validation_batch_size=params['batch'], )

        wandb.finish()
        resultados.append(history.history)
        """       
        history['Product'] = products
        history['Time'] = times
        history['Margen'] = margen   

        #wandb.log({'epochs': epoch,
        #           'loss': np.mean(train_loss),
        #           'acc': float(train_acc),
        #           'val_loss': np.mean(val_loss),
        #           'val_acc': float(val_acc)})
        """
    return resultados