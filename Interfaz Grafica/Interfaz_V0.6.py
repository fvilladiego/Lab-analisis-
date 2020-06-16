import sys
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidget, QTableWidgetItem,QDialog,QMessageBox
from PyQt5 import QtGui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
from scipy import stats
from scipy.optimize import fsolve
from math import e
import seaborn as sns
import xlrd

class programa(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("programa.ui", self)
        self.pos=False
        #botones
        self.graficar.clicked.connect(self.grafica)
        self.aceptar.clicked.connect(self.cantidad)
        self.lineal.clicked.connect(self.reglineal)
        self.slg.clicked.connect(self.regresionslg)
        self.pol.clicked.connect(self.regresionpol)
        self.log.clicked.connect(self.regresionlg)
        self.disb.clicked.connect(self.bernoulli)
        self.histbern.clicked.connect(self.histbernoulli)
        self.disbin.clicked.connect(self.binomial)
        self.histbin.clicked.connect(self.histbinomial)
        self.disexp.clicked.connect(self.exponencial)
        self.histexp.clicked.connect(self.histexponencial)
        self.diship.clicked.connect(self.hipergeometrica)
        self.histhip.clicked.connect(self.histhipergeometrica)
        self.disno.clicked.connect(self.normal)
        self.histno.clicked.connect(self.histnormal)
        self.dispoi.clicked.connect(self.poisson)
        self.histpo.clicked.connect(self.histpoisson)
        self.exc.clicked.connect(self.excel)
        self.exc.clicked.connect(self.errores)
        self.sen.clicked.connect(self.sinusoidal)
    #Datos
    def cantidad(self):
        canti = self.linea.text()

        try:
            canti=int(canti)
        except:

            self.err()
            return None



        self.tableWidget.setRowCount(canti)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(('Datos x', 'Datos y'))
        return canti
    def datos(self):

        lista=[]
        pro=[]

        for x in range(self.cantidad()):
            if self.pos==False:
                try:
                    elementox=self.tableWidget.item(x, 0).text()
                    elementoy = self.tableWidget.item(x, 1).text()
                except:








                    self.error()

                    return None

            if self.pos ==False:
                try:
                    elementox=float(elementox)
                    elementoy = float(elementoy)

                except:
                    self.error()

                    return None

            pro.append(elementox)
            pro.append(elementoy)
            lista.append(pro)
            pro=[]
        if lista==[]:

            return None

        return lista
    def excel(self):
        lista = []
        pro = []
        try:
            ruta =self.ruta.text()
            abrir = xlrd.open_workbook(ruta)
        except:

            return 0
        try:
            sheet = abrir.sheet_by_name(self.hoja.text())
        except:
            self.pos=False
            return 1
        if sheet.nrows<sheet.ncols:
            for i in range(sheet.ncols):
                a = type(sheet.cell_value(0, i))
                b = type(sheet.cell_value(1, i))
                print(lista)
                if a == float:
                    pro.append(sheet.cell_value(0, i))
                    print(lista)
                if b == float:
                    pro.append(sheet.cell_value(1, i))


                else:
                    continue
                lista.append(pro)
                pro = []


        else:
            for i in range(sheet.nrows):
                a = type(sheet.cell_value(i, 0))
                b = type(sheet.cell_value(i, 1))
                if a == float:
                    pro.append(sheet.cell_value(i, 0))

                if b == float:
                    pro.append(sheet.cell_value(i, 1))
                else:
                    continue
                lista.append(pro)
                pro = []
        self.pos=True


        return lista
    def errores(self):
        if self.excel()!=None and self.excel()!=0 and self.excel()!=1:
            self.hojj()
        if self.excel()==0:
            self.rut()
            return None
        if self.excel()==1:
            self.hoj()
            return None
    #Regresiones
    def grafica(self):
        if self.pos==False:
            if self.cantidad() == None:
                self.erdatos()
                return None
            if self.datos() == None:
                return None
            data = np.array(self.datos())
        if self.pos==True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                self.pos=False

                return None
            if self.excel() == []:
                self.cal()
                return None
            data=np.array(self.excel())

        variable1 = np.array(data[:, 0])
        variable2 = np.array(data[:, 1])

        graph_title = "grafica 1"

        graph_x = "x"

        graph_y = "y"

        plt.plot(variable1, variable2, 'b.')
        plt.suptitle(graph_title)
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        plt.grid(b=True)
        plt.legend()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        plt.show()
        plt.close()
    def reglineal(self):
        if self.pos==False:
            if self.cantidad() == None:
                self.erdatos()
                return None
            if self.datos() == None:
                return None
            datos = np.array(self.datos())
        if self.pos==True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                return None
            if self.excel() == []:
                self.cal()
                return None
            datos=np.array(self.excel())
        slope, intercept, r_value, p_value, std_err = stats.linregress(datos[:, 0], datos[:, 1])
        xt = np.linspace(0, datos[-1, -1] + 0.5, 1000)
        rl = xt * slope + intercept
        st = ' Ec. encontrada : y= {}x + {}'.format(slope, intercept)
        self.reglin.setText(st)
        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])
        (xe, ye, out) = (xt, rl, st)
        regraph = True

        graph_title = "Regresion Lineal"
        graph_x = "X"
        graph_y = "Y"

        plt.plot(x, y, 'b.', label='Datos experimentales')
        if regraph:
            plt.plot(xe, ye, 'r', label='Regresion encontrada')
        plt.suptitle(graph_title)
        plt.xlabel(graph_x)
        plt.ylabel(graph_y)
        plt.grid(b=True)
        plt.legend()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        plt.show()
        plt.close()
    def regresionlg(self):
        if self.pos==False:
            if self.cantidad() == None:
                self.erdatos()
                return None
            if self.datos() == None:
                return None
            datos = np.array(self.datos())
        if self.pos==True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                return None
            if self.excel() == []:
                self.cal()
                return None
            datos=np.array(self.excel())
        ly = np.log(datos[:, 1])
        n = np.size(datos[:, 0])
        lx = np.log(datos[:, 0])
        ly_mul_lx = ly * lx
        lx_exp2 = lx ** 2

        def equations(p):
            a, b = p
            return (n * np.log(a) + b * np.sum(lx) - np.sum(ly),
                    np.log(a) * np.sum(lx) + b * np.sum(lx_exp2) - np.sum(ly_mul_lx))

        a, b = fsolve(equations, (1, 1))

        xt = np.linspace(0, datos[-1, -2] + 0.5, 1000)
        rlg = a * (xt) ** b

        out = 'Ec. encontrada : {}x^{}'.format(a, b)
        self.reglg.setText(out)
        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])
        (xe, ye, out)= (xt, rlg, out)
        graph_title = "Regresion Semi-log"

        graph_x = "X"

        graph_y = "Y"

        plt.plot(x, y, 'b.', label='Datos experimentales')

        plt.plot(xe, ye, 'r', label='Regresion encontrada')
        plt.suptitle(graph_title)
        plt.xlabel(graph_x)
        plt.ylabel(graph_y)
        plt.grid(b=True)
        plt.legend()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        plt.show()
        plt.close()
    def regresionslg(self):
        if self.pos==False:
            if self.cantidad() == None:
                self.erdatos()
                return None
            if self.datos() == None:
                return None
            datos = np.array(self.datos())
        if self.pos==True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                return None
            if self.excel() == []:
                self.cal()
                return None
            datos=np.array(self.excel())
        ly = np.log(datos[:, 1])
        x_exp2 = np.square(datos[:, 0])
        x_lny = datos[:, 0] * np.log(datos[:, 1])
        mean_x = np.mean(datos[:, 0])
        mean_ly = np.mean(ly)

        m = ((np.sum(x_lny) - mean_ly * np.sum(datos[:, 0])) / (np.sum(x_exp2) - mean_x * np.sum(datos[:, 0])))

        b = e ** (mean_ly - m * mean_x)

        xt = np.linspace(0, datos[-1, -1] + 0.5, 1000)
        rslg = b * (e) ** (m * xt)
        out = 'Ec. encontrada : {}e^({}x)'.format(b, m)
        self.regslg.setText(out)
        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])
        (xe, ye, out) = (xt, rslg, out)

        graph_title = "Regresion Semi-log"

        graph_x = "X"

        graph_y = "Y"

        plt.plot(x, y, 'b.', label='Datos experimentales')

        plt.plot(xe, ye, 'r', label='Regresion encontrada')
        plt.suptitle(graph_title)
        plt.xlabel(graph_x)
        plt.ylabel(graph_y)
        plt.grid(b=True)
        plt.legend()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        plt.show()
        plt.close()
    def regresionpol(self):
        if self.pos==False:
            if self.cantidad() == None:
                self.erdatos()
                return None
            if self.datos() == None:
                return None
            datos = np.array(self.datos())
        if self.pos==True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                return None
            if self.excel() == []:
                self.cal()
                return None
            datos=np.array(self.excel())
        try:
            g = int(self.grad.text())
        except:
            self.err()
            return None
        n = np.size(datos[:, 0])

        A = np.empty([g + 1, g + 1])
        A[0, 0] = n

        for k in range(1, g + 1, 1):
            t = np.sum(datos[:, 0] ** k)
            i = 0
            j = k
            while (j >= 0 and j <= g and i <= g):
                A[i, j] = t
                i += 1
                j -= 1

        for k in range(g + 1, g * 2 + 1, 1):
            t = np.sum(datos[:, 0] ** k)
            i = g
            j = k - g
            while (j <= g and i <= g):
                A[i, j] = t
                i -= 1
                j += 1

        B = np.empty((g + 1))

        for i in range(0, g + 1):
            l = np.sum(datos[:, 1] * (datos[:, 0] ** i))
            B[i] = l

        sol = np.linalg.solve(A, B)
        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])


        def PolyCoefficients(x, coeffs):
            """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.
            The coefficients must be in ascending order (``x**0`` to ``x**o``).
            """

            o = len(coeffs)

            z = 0
            for i in range(o):
                z += coeffs[i] * x ** i
            return z

        xt = np.linspace(0, datos[-1, -2] + 0.5, 1000)

        p = PolyCoefficients(xt, sol)

        out = 'Coeficientes del polinomio, orden ascendente: {}'.format(sol)
        self.cua.setText(out)
        (xe, ye, out)=(xt, p, out)
        graph_title = "Regresion Polinomial"

        graph_x = "X"

        graph_y = "Y"

        plt.plot(x, y, 'b.', label='Datos experimentales')

        plt.plot(xe, ye, 'r', label='Regresion encontrada')
        plt.suptitle(graph_title)
        plt.xlabel(graph_x)
        plt.ylabel(graph_y)
        plt.grid(b=True)
        plt.legend()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        plt.show()
        plt.close()
    #Modelos de probabilidad
    def bernoulli(self):
        # parametros esteticos de seaborn
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})
        try:
            p=float(self.bern.text())
        except:
            self.err()
            return None
        if p>1 or p<0:
            self.men()
            return None



        bernoulli = stats.bernoulli(p)
        x = np.arange(-1, 3)
        fmp = bernoulli.pmf(x)  # Función de Masa de Probabilidad
        fig, ax = plt.subplots()
        ax.plot(x, fmp, 'bo')
        ax.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)
        ax.set_yticks([0., 0.2, 0.4, 0.6])
        plt.title('Distribución Bernoulli')
        plt.ylabel('probabilidad')
        plt.xlabel('valores')
        plt.show()
    def histbernoulli(self):


        np.random.seed(2016)  # replicar random
        # parametros esteticos de seaborn
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})
        try:
            p=float(self.bern.text())
        except:
            self.err()
            return None
        if p>1 or p<0:
            self.men()
            return None
        bernoulli = stats.bernoulli(p)
        aleatorios = bernoulli.rvs(1000)  # genera aleatorios
        cuenta, cajas, ignorar = plt.hist(aleatorios, 20)

        plt.ylabel('frequencia')
        plt.xlabel('valores')
        plt.title('Histograma Bernoulli')
        plt.show()
    def binomial(self):
        # parametros esteticos de seaborn
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})
        try:
            N = int(self.expbin.text())  # parametros de forma
        except:
            self.err()
            return None
        try:
            p=float(self.probin.text())
        except:
            self.err()
            return None
        if p>1 or p<0:
            self.men()
            return None


        binomial = stats.binom(N, p)  # Distribución
        x = np.arange(binomial.ppf(0.01),
                      binomial.ppf(0.99))
        fmp = binomial.pmf(x)  # Función de Masa de Probabilidad
        plt.plot(x, fmp, '--')
        plt.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)
        plt.title('Distribución Binomial')
        plt.ylabel('probabilidad')
        plt.xlabel('valores')
        plt.show()
    def histbinomial(self):

        np.random.seed(2016)  # replicar random
        # parametros esteticos de seaborn
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})
        try:
            N = int(self.expbin.text())  # parametros de forma
        except:
            self.err()
            return None
        try:
            p=float(self.probin.text())
        except:
            self.err()
            return None
        if p>1 or p<0:
            self.men()
            return None
        binomial = stats.binom(N, p)  # Distribución
        aleatorios = binomial.rvs(1000)  # genera aleatorios
        cuenta, cajas, ignorar = plt.hist(aleatorios, 20)
        plt.ylabel('frequencia')
        plt.xlabel('valores')
        plt.title('Histograma Binomial')
        plt.show()
    def exponencial(self):
        sns.set_palette("deep", desat=.6)  # parametros esteticos de seaborn
        sns.set_context(rc={"figure.figsize": (8, 4)})

        try:
            a = float(self.para.text())
        except:
            self.err()
            return None

        exponencial = stats.expon(a)
        x = np.linspace(exponencial.ppf(0.01),
                        exponencial.ppf(0.99), 100)
        fp = exponencial.pdf(x)  # Función de Probabilidad
        plt.plot(x, fp)
        plt.title('Distribución Exponencial')
        plt.ylabel('probabilidad')
        plt.xlabel('valores')
        plt.show()
    def histexponencial(self):

        np.random.seed(2016)  # replicar random

        # parametros esteticos de seaborn
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})
        try:
            a = float(self.para.text())
        except:
            self.err()
            return None
        exponencial = stats.expon(a)
        aleatorios = exponencial.rvs(1000)  # genera aleatorios
        cuenta, cajas, ignorar = plt.hist(aleatorios, 20)
        plt.ylabel('frequencia')
        plt.xlabel('valores')
        plt.title('Histograma Exponencial')
        plt.show()
    def hipergeometrica(self):
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})

        try:
            M= int(self.hipp.text())  # parametros de forma
        except:
            self.err()
            return None
        try:
            n =  int(self.hipm.text())
        except:
            self.err()
            return None
        try:
            N = int(self.hipc.text())
        except:
            self.err()
            return None

        if n>M:
            self.muestra()
            return None
        if N>M:
            self.favorable()
            return None
        hipergeometrica = stats.hypergeom(M, n, N)  # Distribución
        x = np.arange(0, n + 1)
        fmp = hipergeometrica.pmf(x)  # Función de Masa de Probabilidad
        plt.plot(x, fmp, '--')
        plt.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)
        plt.title('Distribución Hipergeométrica')
        plt.ylabel('probabilidad')
        plt.xlabel('valores')
        plt.show()
    def histhipergeometrica(self):

        np.random.seed(2016)  # replicar random
        # parametros esteticos de seaborn
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})

        try:
            M= int(self.hipp.text())  # parametros de forma
        except:
            self.err()
            return None
        try:
            n =  int(self.hipm.text())
        except:
            self.err()
            return None
        try:
            N = int(self.hipc.text())
        except:
            self.err()
            return None

        if n>M:
            self.muestra()
            return None
        if N>M:
            self.favorable()
            return None
        hipergeometrica = stats.hypergeom(M, n, N)  # Distribución
        aleatorios = hipergeometrica.rvs(1000)  # genera aleatorios
        cuenta, cajas, ignorar = plt.hist(aleatorios, 20)
        plt.ylabel('frequencia')
        plt.xlabel('valores')
        plt.title('Histograma Hipergeométrica')
        plt.show()
    def normal(self):

        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})

        try:
            mu = float(self.desno.text())# media y desviación estándar
        except:
            self.err()
            return None
        try:
            sigma=float(self.deses.text())
        except:
            self.err()
            return None
        if sigma<=0:
            self.negdesv()
            return None


        normal = stats.norm(mu, sigma)
        x = np.linspace(normal.ppf(0.01),
                        normal.ppf(0.99), 100)
        fp = normal.pdf(x)  # Función de Probabilidad
        plt.plot(x, fp)
        plt.title('Distribución Normal')
        plt.ylabel('probabilidad')
        plt.xlabel('valores')
        plt.show()
    def histnormal(self):

        np.random.seed(2016)  # replicar random
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})
        try:
            mu = float(self.desno.text())# media y desviación estándar
        except:
            self.err()
            return None
        try:
            sigma=float(self.deses.text())
        except:
            self.err()
            return None
        if sigma<=0:
            self.negdesv()
            return None
        normal = stats.norm(mu, sigma)
        aleatorios = normal.rvs(1000)  # genera aleatorios
        cuenta, cajas, ignorar = plt.hist(aleatorios, 20)
        plt.ylabel('frequencia')
        plt.xlabel('valores')
        plt.title('Histograma Normal')
        plt.show()
    def poisson(self):
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})

        try:
            mu = float(self.despo.text())  # parametro de forma
        except:
            self.err()
            return None
        if mu<=0:
            self.negativo()
            return None
        poisson = stats.poisson(mu)  # Distribución
        x = np.arange(poisson.ppf(0.01),
                      poisson.ppf(0.99))
        fmp = poisson.pmf(x)  # Función de Masa de Probabilidad
        plt.plot(x, fmp, '--')
        plt.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)
        plt.title('Distribución Poisson')
        plt.ylabel('probabilidad')
        plt.xlabel('valores')
        plt.show()
    def histpoisson(self):

        np.random.seed(2016)  # replicar random
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})
        try:
            mu = float(self.despo.text())  # parametro de forma
        except:
            self.err()
            return None
        if mu<=0:
            self.negativo()
            return None
        poisson = stats.poisson(mu)  # Distribución

        aleatorios = poisson.rvs(1000)  # genera aleatorios
        cuenta, cajas, ignorar = plt.hist(aleatorios, 20)
        plt.ylabel('frequencia')
        plt.xlabel('valores')
        plt.title('Histograma Poisson')
        plt.show()

    def sinusoidal(self):
        g = int(20)
        if self.pos==False:
            if self.cantidad() == None:
                self.erdatos()
                return None
            if self.datos() == None:
                return None
            datos = np.array(self.datos())
        if self.pos==True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                return None
            if self.excel() == []:
                self.cal()
                return None
            datos=np.array(self.excel())
        n = np.size(datos[:, 0])

        A = np.empty([g + 1, g + 1])
        A[0, 0] = n

        for k in range(1, g + 1, 1):
            t = np.sum(datos[:, 0] ** k)
            i = 0
            j = k
            while (j >= 0 and j <= g and i <= g):
                A[i, j] = t
                i += 1
                j -= 1

        for k in range(g + 1, g * 2 + 1, 1):
            t = np.sum(datos[:, 0] ** k)
            i = g
            j = k - g
            while (j <= g and i <= g):
                A[i, j] = t
                i -= 1
                j += 1

        B = np.empty((g + 1))

        for i in range(0, g + 1):
            l = np.sum(datos[:, 1] * (datos[:, 0] ** i))
            B[i] = l

        sol = np.linalg.solve(A, B)

        solu = sol
        xt = np.linspace(0, datos[-1, -2] + 0.5, 1000)
        k = 0
        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])
        for i in range(1, 4):
            if i % 2 != 0:
                if abs(solu[i]) > abs(solu[i + 1]):
                    k += 1
                else:
                    k -= 1
            else:
                if abs(solu[i]) < abs(solu[i + 1]):
                    k += 1
                else:
                    k -= 1

        if k > 0:  # Seno
            c = solu[0]
            w = np.sqrt((abs(solu[3]) * 6) / (solu[1]))
            A = solu[1] / w

            out = '{}sin({}x)+{}'.format(A, w, c)
            self.sin.setText(out)
            seno = A * np.sin(w * xt) + c
            plt.plot(x, y)

            graph_title = "Regresión sinusoidal"
            graph_x = "X"
            graph_y = "Y"
            plt.plot(x, y, 'b.', label='Datos experimentales')
            plt.plot(xt, seno, label=out)
            plt.suptitle(graph_title)
            plt.xlabel(graph_x)
            plt.ylabel(graph_y)
            plt.grid(b=True)
            plt.legend()
            plt.show()
            plt.close()

        else:  # Coseno

            w = np.sqrt((abs(solu[4]) * 24) / (2 * abs(solu[2])))

            A = (abs(solu[2]) * 2) / (w ** 2)

            c = solu[0] - A

            out = print('{}cos({}x)+{}'.format(A, w, c))
            self.sin.setText(out)
            coseno = A * np.cos(w * xt) + c

            plt.plot(x, y)
            graph_title = "Regresion sinusoidal"
            graph_x = "X"
            graph_y = "Y"
            plt.plot(x, y, 'b.', label='Datos experimentales')
            plt.plot(xt, coseno, label=out)
            plt.suptitle(graph_title)
            plt.xlabel(graph_x)
            plt.ylabel(graph_y)
            plt.grid(b=True)
            plt.legend()
            plt.show()
            plt.close()

    #Errores
    def error(self):
        msg=QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("No ha introducido valores correctos en todas las celdas de la tabla")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def incompleto(self):
        msg=QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("Por favor rellene toda la tabla")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def err(self):
        msg=QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("No ha ingresado un caracter valido")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def men(self):
        msg=QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("La probabilidad debe estar entre cero y uno")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def erdatos(self):
        msg=QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("No ha ingresado cuantos datos quiere")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def muestra(self):
        msg = QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("El tamaño de la muestra debe ser menor que el tamaño de la poblacion")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def favorable(self):
        msg = QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("Los casos favorables deben ser menores que el tamaño de la poblacion")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def negativo(self):
        msg = QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("El parametro debe ser mayor a cero")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def negdesv(self):
        msg = QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("La desviacion estandar debe ser mayor a cero")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def rut(self):
        msg = QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("Error en la ruta del archivo")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def hoj(self):
        msg = QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("No existe una hoja con ese nombre")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def hojj(self):

        msg = QMessageBox()
        msg.setWindowTitle("Hoja cargada")
        msg.setText("Hoja cargada exitosamente")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()
    def cal(self):
        msg = QMessageBox()
        msg.setWindowTitle("Hoja cargada")
        msg.setText("No se reconocen los datos de la hoja de calculo")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
if __name__ == "__main__":

    prog = QApplication(sys.argv)
    GUI = programa()
    GUI.show()
    sys.exit(prog.exec_())