#Manejo de Datos
import pandas as pd


#Librerias estandar (Extras)
import os
import time


# Devuelve una lista con lo indices que no se encontraron lso archivos y el producto
# Servira para ver si se teinen todas los frames de la fecha
def comprobarFrames(dfOrignial, path_base, products, times,dataType, delete=1):
    start_time = time.time()

    dfTotal = pd.unique(dfOrignial['fecha'])

    no_fecha = []
    for fecha in dfTotal:
        year, month, day, hour = fecha.split('-')

        existe = True
        for p in products:
            for t in range(len(times)):
                if dataType == 'csv':
                    filename = f'{path_base}comprimido/{fecha}/{fecha}_{p}_{t}.csv'
                if dataType == 'png':
                    filename = f'{path_base}PNG/{fecha}/{fecha}_{t}.png'
                existe = os.path.exists(filename)
                if not existe:
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
