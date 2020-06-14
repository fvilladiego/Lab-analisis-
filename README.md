
# Lab-analisis-v.0.1

Lab-analisis es un programa que facilita el analisis de laboratorios gracias a el número de herramientas disponibles en él. La prioridad del programa es ahorrar tiempo al usuario y que este pueda rápidamente sacar el mejor provecho de sus datos. Regresiones, Distribuciones , gráficas y más. 

## Empezando
El programa fue desarrollado en Python 3.7, en múltiples plataformas por lo cúal es poco probable que ocurran conflictos de hardware.

### Prerequisitos
Python 3.7

Librerías necesarias para el funcionamiento del programa:

```
scipy
matplotlib
numpy
pandas
PyQt5
Sys
xlrd
```
Se recomienda usar la última versión de estas.

### Instalación

A continuación la lista de pasos para la correcta instalación del programa:

1. Instalar las librerias correspondientes, en la sección anterior.

2. Descargar la carpeta 'Interfaz-grafica' y ponerla en la ubicación deseada.

## Correr el programa

Para correr el programa se debe verificar que la carpeta de la sección anterior contenga el archivo .py y .ui. Entonces se debe correr el archivo.py y, si los prerrequisitos se cumplen el programa debe de funcionar correctamente.

### Herramientas
## Ingresar datos 

Lab-analisis ofrece varias herramientas. Ventana izquierda superior le da dos opciones al usuario para ingresar sus datos: una manual , y una ingresando un excel, con su respectiva ruta dentro del sistema y Hoja de cálculo. Para el ingreso de datos manuales se debe especificar en la parte superior izquierda el número de datos que desea ingresar (Entiéndase datos por 'parejas de datos'). Se hace la aclaración que el programa espera dos columnas solamente, correspondientes a los datos de el eje 'x' en la columna izquierda, y 'y' en la columna derecha.

## Regresiones

El programa permite hacer 5 tipos de regresiones: Lineal, Exponencial(Semi-Logarítmica), Logarítmica, Polinomial y Sinusoidal(no incluida en la actual versión).
Para las primeras 3 solo basta hacer click en la ventana correspondiente y el programa devolverá la función encontrada, junto a una gráfica de los datos con estos valores. Para la regresión polinomial es necesario primero especificar el número de grados del polinomio que se espera recibir. Para mayor precisión es recomendable usar un número de grados prudente (20 es una muy buena opción).
## Modelos de probabilidad usuales

Lab-analisis permite al usuario usar 6 tipos de Distribuciones de probabilidad: Bernoulli, Binomial, Exponencial, Hipergeométrica, Normal y Poisson. Estas se pueden encontrar en la parte inferior del programa, primera pestaña. Cada uno de esos modelos contiene ventanas para ingresar los datos correspondientes, los cuales dependen de cada una. En la parte derecha de esta ventana, se ofrece la opción de hacer gráficas de las distribuciones.


## Creado en:

*Python

## Para contribuyentes

Contacto directamente por Github.

## Versión actual

v.0.1

## Autores

* **Juan Bustos** - *Distribuciones y general* - [nombre1](link)
* **Daniel Pineda** - *Interfaz gráfica y general* - [nombre2](link)
* **Federico Villadiego** - *Regresiones y general* - [fvilladiego](link)

Véase también la lista de contribuyentes (link) que participaron en este proyecto.

## Licencia

Por definir.

## Agradecimientos

* A toda la comunidad freeware de python e internet, nuestros compañeros de la clase de Programación y nuestro profesor R.Amezquita. 


