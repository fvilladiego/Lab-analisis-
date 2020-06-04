import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Aquí es importante saber qué datos se quiere gráficar, entonces hago un condicional
#Variables que se necesitan antes de iniciar este código: datos_import , ruta excel, data.
#def obtener_datos(datos_import=False):

if datos_import == True:  #La idea es que esta variable sea una opción en la plataforma
    data_pd = pd.ExcelFile(ruta_excel)
    datar = pd.read_excel(data_pd)
    data = datar.to_numpy()           #data es la matriz con los datos. Si no se importa excel entonces debe estar definida en el programa
    dimesiones = data.shape
        
#        return (data)
        
#    else:
    
#        return (data)   #Esta sería la que ya está definida resultado del ingreso de datos del usuario
    

variable1 = np.array(data[:,0])
variable2 = np.array(data[:,1])




graph_title=input('Ingrese el título de su gráfica: ')

graph_x=input('Ingrese el título del eje x: ')

graph_y=input('Ingrese título del eje y: ')

exportar=input("Desea exportar la gráfica cómo imagen?")

if exportar==True:
    file_name = input("Nombre para esta imagen (evitar espacios y tildes): ")
    
plt.plot(variable1,variable2 ,'b.')            
plt.suptitle(graph_title)
plt.xlabel(variable1)
plt.ylabel(variable2)
plt.grid(b=True)
plt.legend()
if exportar == True:
    plt.savefig(file_name)

plt.show()
plt.close()

