import sys
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidget, QTableWidgetItem
from PyQt5 import QtGui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
from scipy import stats
from scipy.optimize import fsolve
from math import e
import seaborn as sns

class programa(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("programa.ui", self)
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

    def cantidad(self):
        canti = self.linea.text()
        canti=int(canti)
        self.tableWidget.setRowCount(canti)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(('Datos x', 'Datos y'))
        return canti
    def datos(self):

        lista=[]
        pro=[]
        for x in range(self.cantidad()):

            elementox=self.tableWidget.item(x, 0).text()
            elementox=float(elementox)
            pro.append(elementox)

            elementoy=self.tableWidget.item(x, 1).text()
            elementoy=float(elementoy)
            pro.append(elementoy)

            lista.append(pro)
            pro=[]

        return lista

    def grafica(self):

        data = np.array(self.datos())

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
        datos = np.array(self.datos())
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
        datos = np.array(self.datos())
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
        datos = np.array(self.datos())
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
        datos = np.array(self.datos())
        g = int(self.grad.text())
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
    def bernoulli(self):
        # parametros esteticos de seaborn
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})

        p=float(self.bern.text())
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
        p = float(self.bern.text())
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

        N, p = int(self.expbin.text()), float(self.probin.text())  # parametros de forma
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
        N, p = int(self.expbin.text()), float(self.probin.text() ) # parametros de forma
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

        a = float(self.para.text())
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
        a = float(self.para.text())
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

        M, n, N = int(self.hipp.text()), int(self.hipm.text()), int(self.hipc.text())  # parametros de forma
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

        M, n, N = int(self.hipp.text()), int(self.hipm.text()), int(self.hipc.text())  # parametros de forma
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
        mu, sigma = float(self.desno.text()), float(self.deses.text())  # media y desviación estándar
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
        mu, sigma = float(self.desno.text()), float(self.deses.text())  # media y desviación estándar
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
        mu = float(self.despo.text())  # parametro de forma
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
        mu = float(self.despo.text())  # parametro de forma
        poisson = stats.poisson(mu)  # Distribución

        aleatorios = poisson.rvs(1000)  # genera aleatorios
        cuenta, cajas, ignorar = plt.hist(aleatorios, 20)
        plt.ylabel('frequencia')
        plt.xlabel('valores')
        plt.title('Histograma Poisson')
        plt.show()
if __name__ == "__main__":

    prog = QApplication(sys.argv)
    GUI = programa()
    GUI.show()
    sys.exit(prog.exec_())

