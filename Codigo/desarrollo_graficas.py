import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from scipy.optimize import fsolve
from math import e

##Este es codigo correspondiente a las regresiones, escritas como objetos de la clase regresion.


class regresion:
    
    def __init__(self,datos):
        self.datos=datos
        
    def regresion_l(datos):
        slope, intercept, r_value, p_value, std_err  = stats.linregress( datos[:,0] , datos[:,1] )
        xt= np.linspace(0 , datos[-1,-1] + 0.5, 1000 )
        rl = xt*slope + intercept
        st =' Ec. encontrada : y= {}x + {}'.format(slope, intercept)
        return (xt,rl,st)

    def regresion_slg(datos):
         ly=np.log(datos[:,1])
         x_exp2=np.square(datos[:,0])
         x_lny=datos[:,0]*np.log(datos[:,1])
         mean_x= np.mean(datos[:,0])
         mean_ly=np.mean(ly)
         
         
         m=((np.sum(x_lny)- mean_ly*np.sum(datos[:,0]))/(np.sum(x_exp2) - mean_x*np.sum(datos[:,0])))
         
         b=e**(mean_ly - m*mean_x)
         
         xt= np.linspace(0 , datos[-1,-1] + 0.5, 1000 )
         rslg = b*(e)**(m*xt)
         out = 'Ec. encontrada : {}e^({}x)'.format(b,m)
         
         return(xt,rslg,out)
         
    def regresion_lg(datos):
        ly=np.log(datos[:,1])
        n=np.size(datos[:,0])
        lx=np.log(datos[:,0])
        ly_mul_lx=ly*lx
        lx_exp2=lx**2
        
        def equations(p):
            a, b = p
            return(n*np.log(a)+b*np.sum(lx) - np.sum(ly) , np.log(a)*np.sum(lx)+b*np.sum(lx_exp2) - np.sum(ly_mul_lx))
            
        a , b = fsolve(equations, (1,1))
        
        
        xt= np.linspace(0 , datos[-1,-2] + 0.5, 1000 )
        rlg = a*(xt)**b
        
        out = 'Ec. encontrada : {}x^{}'.format(a,b)
        
        return(xt,rlg,out)
        
    def regresion_pol(datos):
        
        g=int(input('Cuantos grados cree que tiene su polinomio?'))
        n=np.size(datos[:,0])
        
        A=np.empty([g+1,g+1])
        A[0,0]=n

        
        for k in range(1,g+1,1):
            t=np.sum(datos[:,0]**k)
            i=0
            j=k
            while (j >= 0 and j <= g and i <= g) :
                A[i,j]=t
                i +=1
                j -=1
                
                
        for k in range(g+1,g*2+1,1):
            t=np.sum(datos[:,0]**k)
            i=g
            j=k-g
            while (j <= g and i <= g):
                A[i,j]=t
                i-=1
                j+=1
                
        
        B=np.empty((g+1))
        
        for i in range(0,g+1):
            l=np.sum(datos[:,1]*(datos[:,0]**i))
            B[i]=l
            
           
        sol= np.linalg.solve(A,B)
        
        def PolyCoefficients(x, coeffs):
            """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

            The coefficients must be in ascending order (``x**0`` to ``x**o``).
            """
            o = len(coeffs)
            
            z = 0
            for i in range(o):
                z += coeffs[i]*x**i
            return z
        
        xt= np.linspace(0 , datos[-1,-2] + 0.5, 1000 )
        
        p=PolyCoefficients(xt,sol)
        
        out = 'Coeficientes del polinomio, orden ascendente:  {}'.format(sol)
        
        return(xt,p,out )
    
    
    def sinusoidal(self): #Se debe aclarar al usuario que esta regresión es útil para puntos cercanos al 0, la función coseno tiene un error grande
    solu =regresion.regresion_pol() #Se debe añadir en el código de interfaz que regresion.regresion_pol devuelva el valor sol, correspondiente a los coeficientes
    
    k=0
    for i in range(1,4):
        if i%2 != 0:
            if abs(solu[i]) > abs(solu[i+1]):
                k +=1
            else:
                k -=1
        else:
            if abs(solu[i])< abs(solu[i+1]):
                k +=1
            else: 
                k -=1
                
    if k > 0: #Seno
        c = solu[0]
        w = np.sqrt((abs(solu[3])*6)/(solu[1])) 
        A = solu[1]/w
        
        out =print('{}sin({}x)+{}'.format(A , w ,c))
       
        
    else: #Coseno
        
        w =  np.sqrt((abs(solu[4])*24) / (2*abs(solu[2]))) 
        
        A = (abs(solu[2])*2) / (w**2)
        
        c = solu[0] - A

        out =print('{}cos({}x)+{}'.format(A , w ,c))
         
    return out
          
        
lista_regresion= {'L' : regresion.regresion_l, 'SLG' : regresion.regresion_slg , 'LG' : regresion.regresion_lg , 'POL' : regresion.regresion_pol}


#Aquí es importante saber qué datos se quiere gráficar, entonces hago un condicional
#Variables que se necesitan antes de iniciar este código: datos_import , ruta excel, data.
#def obtener_datos(datos_import=False):


if datos_import == True:  #La idea es que esta variable sea una opción en la plataforma
    data_pd = pd.ExcelFile(ruta_excel)
    datar = pd.read_excel(data_pd)
    data = datar.to_numpy()           #data es la matriz con los datos. Si no se importa excel entonces debe estar definida en el programa
    dimesiones = data.shape
        

        

    

    

x = np.array(data[:,0])
y = np.array(data[:,1])


ask_regresion= input('Desea hacer una regresion de sus datos? ')

if ask_regresion:
    a=input("Para que tipo de curva quiere su regresion? L -> Lineal  ;  POL -> polinomial ; LG -> logaritmica ; SLG -> Semi logaritmica :  ")

    (xe , ye , out) = lista_regresion[a](data)
    
    print(out)
    
    regraph=True

graph_title=input('Ingrese el título de su gráfica: ')

graph_x=input('Ingrese el título del eje x: ')

graph_y=input('Ingrese título del eje y: ')

exportar=input("Desea exportar la gráfica cómo imagen? ")

if exportar==True:
    file_name = input("Nombre para esta imagen (evitar espacios y tildes): ")
    
plt.plot(x , y,'b.',label='Datos experimentales')   
if regraph:
    plt.plot(xe,ye,'r', label='Regresion encontrada')         
plt.suptitle(graph_title)
plt.xlabel(graph_x)
plt.ylabel(graph_y)
plt.grid(b=True)
plt.legend()
if exportar == True:
    plt.savefig(file_name)

plt.show()
plt.close()
