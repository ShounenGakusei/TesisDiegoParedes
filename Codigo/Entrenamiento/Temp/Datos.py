#Manejo de Datos
import os
import pandas as pd



#Machine learning
from sklearn.utils import shuffle

#Librerias estandar (Extras)
import time


#Del dataset guardamos los datos mas importantes en una columna para facilitar su lectura
def obtenerDir(row):
    fecha = row['fecha']

    year, month, day, hour = fecha.split('-')
    # filename = f'{path_base}comprimido/{year}_{month}_{day}/{hour}/'
    return f"{row['XO']}--{row['XA']}--{fecha}"

# Lee el archivo "filename" de datos de precipitacion y
# regresa un df que facilite la lectura del dataset para el entrenmaiento
def obtenerDatos(filename):
    start_time = time.time()
    pdata = pd.read_csv(filename)

    # Quitamos los valores NA
    pdata = pdata[pdata['dato'].notna()]

    # Definimos un solo tipo (str) pora asi poder convertirlo a tensor
    pdata = pdata.astype({"dato": str, "XO": str, "XA": str, "fecha": str})

    # Definimos la nueva columna para guardar el XO, XA y fecha
    pdata['imagen'] = pdata.apply(obtenerDir, axis=1)

    # Seleccionamos solo las columnas necesarias :
    # precipitacion, Estacion (Longitud), Estacion (Latitud), Fecha (año-mes-dia-hora)
    # pdataX = pdata.loc[:, ['dato','umbral','altura', 'imagen', 'fecha']]
    pdata = pdata.astype({"dato": str, "umbral": str, "altura": str, "imagen": str, "fecha": str})

    # Barajeamos los datos
    pdata = shuffle(pdata)

    print(f'{len(pdata)} datos leidos')
    print("Tiempo tomado en leer datos: %.2fs" % (time.time() - start_time))
    return pdata


# Devuelve una lista con lo indices que no se encontraron lso archivos y el producto
# Servira para ver si se teinen todas los frames de la fecha
def comprobarFrames(dfOrignial, path_base, products, times, delete=1):
    # dfOrignial = obtenerDatos(datafile)

    start_time = time.time()

    dfTotal = pd.unique(dfOrignial['fecha'])
    no_fecha = []
    for fecha in dfTotal:
        year, month, day, hour = fecha.split('-')
        existe = True
        for p in products:
            for t in range(len(times)):
                filename = f'{path_base}PNG/{fecha}/{fecha}_{t}.png'
                try:
                    file_size = os.path.getsize(filename)
                    existe = file_size > 4100000
                except:
                    existe = False
                    break

            if not existe:
                break
        if not existe:
            no_fecha.append(fecha)

    if delete:
        antes = len(dfOrignial)
        df2 = dfOrignial[~dfOrignial['fecha'].isin(no_fecha)]
        despues = len(df2)
        print(f'{antes - despues}/{antes} datos eliminados: No se encontraron los archivos de imagenes satelitales')
    else:
        df2 = dfOrignial

    print("Tiempo tomado en verificar datos: %.2fs" % (time.time() - start_time))
    return df2, no_fecha


def limpiarDatos(listNames, path_imagenes, products, times, delete=1):
    df = []
    start_time = time.time()
    print(f'Se leera los archivos de datasets...')
    for name in listNames:
        try:
            df.append(pd.read_csv(name))
        except:
            print(f'No se pudo leer el archivo {name} de dataset')
            return -1

    dsCompleto = pd.concat(df, ignore_index=True)
    print("Tiempo tomado: %.2fs" % (time.time() - start_time))
    print(f'+Cantidad de datos leidos {len(dsCompleto)}')

    # Quitamos los NA valores
    print(f'Se elimnara los valores nulos')
    dsCompleto.dropna(subset=['dato'], axis='index', inplace=True)
    dsCompleto = dsCompleto[dsCompleto['flag'] != 'ND']
    print("Tiempo tomado: %.2fs" % (time.time() - start_time))
    print(f'+Cantidad de datos luego de elimnar nulos {len(dsCompleto)}')

    # Buscamos imagenes satelitales para lso archivos
    print(f'Se buscara las imagenes satelitales para los datos...')
    dfImagenes, no_fecha = comprobarFrames(dsCompleto, path_imagenes, products, times, delete)
    print("Tiempo tomado: %.2fs" % (time.time() - start_time))

    # Agregamos lso datos de las estaciones al dataset
    print(f'Se agregara los datos de las estaciones(cordenadas, umbral)...')
    dfImagenes['imagen'] = dfImagenes.apply(obtenerDir, axis=1)
    print("Tiempo tomado: %.2fs" % (time.time() - start_time))
    print(f'+Cantidad Final de datos total {len(dfImagenes)}')
    return shuffle(dfImagenes), no_fecha