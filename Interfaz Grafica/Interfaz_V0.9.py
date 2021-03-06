import sys
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidget, QTableWidgetItem,QDialog,QMessageBox,QSlider
from PyQt5 import QtGui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
from scipy import stats
from scipy.optimize import fsolve
from math import e
import xlrd
import os
import errno
import math
class programa(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("programa.ui", self)
        self.pos=False
        try:
            os.mkdir('graficas')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.lin=False
        self.lg=False
        self.sg=False
        self.pl=False
        self.sn=False
        self.grd=False
        #graficar
        self.grl=False
        self.explin = False
        self.grlg=False
        self.explg=False
        self.grsg=False
        self.expsg = False
        self.grpol=False
        self.exppl = False
        self.grpun = False
        self.exppun = False
        self.grsn=False
        self.expsn = False

        #exportar distribuciones
        self.grabn=False
        self.expbr = False
        self.hgrabn=False
        self.hexpbr=False

        self.grabi=False
        self.expbi=False
        self.hgrabi=False
        self.exphbi=False

        self.grapoi = False
        self.exppoi = False
        self.hgrapoi = False
        self.exphpoi = False

        self.grano = False
        self.expno = False
        self.hgrano = False
        self.exphno = False

        self.grahi = False
        self.exphi = False
        self.hgrahi = False
        self.exphhi = False

        self.grapo = False
        self.exppo = False
        self.hgrapo = False
        self.exphpo = False

        self.graun = False
        self.expun = False
        self.hgraun = False
        self.exphun = False

        #botones
        self.exporthun.clicked.connect(self.exporhun)
        self.exportun.clicked.connect(self.exporun)
        self.unif.clicked.connect(self.grafun)
        self.exporthpo.clicked.connect(self.exporhpo)
        self.histpo.clicked.connect(self.grafhpo)
        self.dispoi.clicked.connect(self.grafpoi)
        self.histhip.clicked.connect(self.hghip)
        self.diship.clicked.connect(self.grafhip)
        self.histno.clicked.connect(self.histgrano)
        self.disno.clicked.connect(self.grafno)
        self.histexp.clicked.connect(self.hgrafexp)
        self.disexp.clicked.connect(self.grafexp)
        self.histbin.clicked.connect(self.grahbin)
        self.disbin.clicked.connect(self.grabin)
        self.graficar.clicked.connect(self.grapunt)
        self.graficar.clicked.connect(self.grafica)
        self.aceptar.clicked.connect(self.cantidad)
        self.lineal.clicked.connect(self.gralin)
        self.lineal.clicked.connect(self.reglineal)
        self.slg.clicked.connect(self.graslog)
        self.slg.clicked.connect(self.regresionslg)
        self.pol.clicked.connect(self.grapol)
        self.pol.clicked.connect(self.regresionpol)
        self.log.clicked.connect(self.gralog)
        self.log.clicked.connect(self.regresionlg)

        self.expberno.clicked.connect(self.expber)
        self.disb.clicked.connect(self.bernoulli)
        self.exportbrn.clicked.connect(self.hexpbern)
        self.histbern.clicked.connect(self.histbernoulli)
        self.exporbi.clicked.connect(self.exporbin)
        self.disbin.clicked.connect(self.binomial)
        self.exportbin.clicked.connect(self.exphbin)
        self.histbin.clicked.connect(self.histbinomial)
        self.exportex.clicked.connect(self.exporexp)
        self.disexp.clicked.connect(self.exponencial)
        self.exporthex.clicked.connect(self.hexporexp)
        self.exporthno.clicked.connect(self.histexpno)
        self.histexp.clicked.connect(self.histexponencial)
        self.exporthi.clicked.connect(self.exporhip)
        self.diship.clicked.connect(self.hipergeometrica)
        self.exporthhip.clicked.connect(self.hexporhip)
        self.histhip.clicked.connect(self.histhipergeometrica)
        self.exportno.clicked.connect(self.exprnon)
        self.disno.clicked.connect(self.normal)
        self.histno.clicked.connect(self.histnormal)
        self.exportpo.clicked.connect(self.expopoi)
        self.dispoi.clicked.connect(self.poisson)
        self.histpo.clicked.connect(self.histpoisson)
        self.exc.clicked.connect(self.excel)
        self.exc.clicked.connect(self.errores)
        self.sen.clicked.connect(self.grasin)
        self.sen.clicked.connect(self.sinusoidal)
        self.unif.clicked.connect(self.uniforme)
        self.hunif.clicked.connect(self.grafhun)
        self.aut.clicked.connect(self.re_auto)
        self.lingr.clicked.connect(self.expl)
        self.lingr.clicked.connect(self.reglineal)
        self.exportarlg.clicked.connect(self.explog)
        self.exportarlg.clicked.connect(self.regresionlg)
        self.exportarsg.clicked.connect(self.expslog)
        self.exportarsg.clicked.connect(self.regresionslg)
        self.exportarpl.clicked.connect(self.exppol)
        self.exportarpl.clicked.connect(self.regresionpol)
        self.exportarpn.clicked.connect(self.exppunt)
        self.exportarpn.clicked.connect(self.grafica)
        self.exportarsn.clicked.connect(self.expseno)
        self.exportarsn.clicked.connect(self.sinusoidal)

        self.disb.clicked.connect(self.grabern)
        self.histbern.clicked.connect(self.hgrabrn)
        #probabilidades
        self.calsimp.clicked.connect(self.simple)
        self.calper.clicked.connect(self.permutacion)
        self.calcom.clicked.connect(self.combinaciones)
        self.valr.clicked.connect(self.valor)
        self.calbino.clicked.connect(self.pbinomial)
        self.calgeo.clicked.connect(self.pgeometrica)
        self.calhip.clicked.connect(self.phiper)
        self.calpoi.clicked.connect(self.ppoisson)
        self.calexp.clicked.connect(self.pexponencial)
        self.calnorm.clicked.connect(self.pnormal)
        #deslizador colores regresiones
        self.rojop.valueChanged.connect(self.punrojo)
        self.azulp.valueChanged.connect(self.punazul)
        self.verdep.valueChanged.connect(self.punverde)
        self.srojo.valueChanged.connect(self.rojo)
        self.sazul.valueChanged.connect(self.azul)
        self.sverde.valueChanged.connect(self.verde)
        self.linrojo.valueChanged.connect(self.linealrojo)
        self.linazul.valueChanged.connect(self.linealazul)
        self.linverde.valueChanged.connect(self.linealverde)
        self.logrojo.valueChanged.connect(self.logarojo)
        self.logverde.valueChanged.connect(self.logaverde)
        self.logazul.valueChanged.connect(self.logaazul)
        self.logrojo.valueChanged.connect(self.logarojo)
        self.logverde.valueChanged.connect(self.logaverde)
        self.logazul.valueChanged.connect(self.logaazul)
        self.lorojo.valueChanged.connect(self.glogrojo)
        self.loverde.valueChanged.connect(self.glogverde)
        self.loazul.valueChanged.connect(self.glogazul)
        self.semirojo.valueChanged.connect(self.psemirojo)
        self.semiverde.valueChanged.connect(self.psemiverde)
        self.semiazul.valueChanged.connect(self.psemiazul)
        self.semrojo.valueChanged.connect(self.gsemirojo)
        self.semverde.valueChanged.connect(self.gsemiverde)
        self.semazul.valueChanged.connect(self.gsemiazul)
        self.polrojo.valueChanged.connect(self.ppolrojo)
        self.polverde.valueChanged.connect(self.ppolverde)
        self.polazul.valueChanged.connect(self.ppolazul)
        self.porojo.valueChanged.connect(self.gpolrojo)
        self.poverde.valueChanged.connect(self.gpolverde)
        self.poazul.valueChanged.connect(self.gpolazul)
        # deslizador colores distribuciones e histogramas
        self.berrojo.valueChanged.connect(self.bernrojo)
        self.berverde.valueChanged.connect(self.bernverde)
        self.berazul.valueChanged.connect(self.bernazul)
        self.hbrojo.valueChanged.connect(self.hbernrojo)
        self.hbverde.valueChanged.connect(self.hbernverde)
        self.hbazul.valueChanged.connect(self.hbernazul)
        self.binrojo.valueChanged.connect(self.binorojo)
        self.binverde.valueChanged.connect(self.binoverde)
        self.binazul.valueChanged.connect(self.binoazul)
        self.hbirojo.valueChanged.connect(self.hbinorojo)
        self.hbiverde.valueChanged.connect(self.hbinoverde)
        self.hbiazul.valueChanged.connect(self.hbinoazul)
        self.exrojo.valueChanged.connect(self.exprojo)
        self.exverde.valueChanged.connect(self.expverde)
        self.exazul.valueChanged.connect(self.expazul)
        self.herojo.valueChanged.connect(self.hexprojo)
        self.heverde.valueChanged.connect(self.hexpverde)
        self.heazul.valueChanged.connect(self.hexpazul)
        self.norrojo.valueChanged.connect(self.normalrojo)
        self.norverde.valueChanged.connect(self.normalverde)
        self.norazul.valueChanged.connect(self.normalazul)
        self.hnrojo.valueChanged.connect(self.hnormalrojo)
        self.hnverde.valueChanged.connect(self.hnormalverde)
        self.hnazul.valueChanged.connect(self.hnormalazul)
        self.hiprojo.valueChanged.connect(self.hiperrojo)
        self.hipverde.valueChanged.connect(self.hiperverde)
        self.hipazul.valueChanged.connect(self.hiperazul)
        self.hhrojo.valueChanged.connect(self.hhiperrojo)
        self.hhverde.valueChanged.connect(self.hhiperverde)
        self.hhazul.valueChanged.connect(self.hhiperazul)
        self.poirojo.valueChanged.connect(self.poisrojo)
        self.poiverde.valueChanged.connect(self.poisverde)
        self.poiazul.valueChanged.connect(self.poisazul)
        self.hprojo.valueChanged.connect(self.hpoisrojo)
        self.hpverde.valueChanged.connect(self.hpoisverde)
        self.hpazul.valueChanged.connect(self.hpoisazul)
        self.sirojo.valueChanged.connect(self.senrojo)
        self.siverde.valueChanged.connect(self.senverde)
        self.siazul.valueChanged.connect(self.senazul)
        self.serojo.valueChanged.connect(self.gsenrojo)
        self.severde.valueChanged.connect(self.gsenverde)
        self.seazul.valueChanged.connect(self.gsenazul)
        self.unrojo.valueChanged.connect(self.unifrojo)
        self.unverde.valueChanged.connect(self.unifverde)
        self.unazul.valueChanged.connect(self.unifazul)
        self.hunrojo.valueChanged.connect(self.hiunifrojo)
        self.hunverde.valueChanged.connect(self.hiunifverde)
        self.hunazul.valueChanged.connect(self.hiunifazul)


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

                if a == float:
                    pro.append(sheet.cell_value(0, i))

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
    def grapunt(self):
        self.grpun=True
    def gralin(self):
        self.lin=True
        self.grl=True
    def gralog(self):
        self.lg=True
        self.grlg=True
    def graslog(self):
        self.sg=True
        self.grsg = True
    def grapol(self):
        self.pl=True
        self.grpol=True
    def grasin(self):
        self.sn=True
        self.grsn=True
    def expl(self):
        self.explin = True
        self.lin = True
    def explog(self):
        self.lg = True
        self.explg=True
    def expslog(self):
        self.sg = True
        self.expsg=True
    def exppol(self):
        self.pl=True
        self.exppl=True
    def exppunt(self):
        self.exppun=True
    def expseno(self):
        self.sn = True
        self.expsn=True
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

        graph_title = self.nombre.text()

        graph_x = self.ejex.text()

        graph_y = self.ejey.text()

        plt.scatter(variable1, variable2, color=(self.punrojo(),self.punverde(),self.punazul()))
        plt.suptitle(graph_title)
        plt.xlabel(graph_x)
        plt.ylabel(graph_y)
        plt.grid(b=True)
        plt.legend()
        if self.exppun== True:
            plt.savefig("graficas/" + graph_title)
            plt.close()
            self.gu()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        if self.grpun == True:

            plt.show()
            plt.close()
        self.grpun = False

    def reglineal(self):
        if self.pos==False:
            if self.cantidad() == None:
                self.erdatos()
                self.lin =False
                return None
            if self.datos() == None:
                self.lin =False
                return None
            datos = np.array(self.datos())
        if self.pos==True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                self.lin = False
                return None
            if self.excel() == []:
                self.cal()
                self.lin = False
                return None
            datos=np.array(self.excel())
        slope, intercept, r_value, p_value, std_err = stats.linregress(datos[:, 0], datos[:, 1])
        xt = np.linspace(0, datos[-1, -1] + 0.5, 1000)
        rl = xt * slope + intercept
        st = ' Ec. encontrada : y= {}x + {}'.format(slope, intercept)

        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])
        (xe, ye, out) = (xt, rl, st)

        if self.lin==True:

            graph_title = self.nlineal.text()
            graph_x = self.xlineal.text()
            graph_y = self.ylineal.text()

            plt.scatter(x, y, color=(self.linealrojo(),self.linealverde(),self.linealazul()), label='Datos experimentales')

            plt.plot(xe, ye, color=(self.rojo(),self.verde(),self.azul()), label='Regresion encontrada')
            plt.suptitle(graph_title)
            plt.xlabel(graph_x)
            plt.ylabel(graph_y)
            plt.grid(b=True)
            plt.legend()
            if self.explin==True:
                plt.savefig("graficas/"+graph_title)
                plt.close()
                self.gu()
            self.explin = False
            pyplot.axhline(0, color="black")
            pyplot.axvline(0, color="black")

            if self.grl==True:
                self.reglin.setText(st)
                plt.show()
                plt.close()
            self.grl=False
            self.lin=False
        yp = x*slope + intercept

        R2 = (np.var(yp))/(np.var(y))
        return R2
    def regresionlg(self):
        if self.pos==False:
            if self.cantidad() == None:
                self.erdatos()
                self.lg = False
                return None
            if self.datos() == None:
                self.lg = False
                return None
            datos = np.array(self.datos())
        if self.pos==True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                self.lg = False
                return None
            if self.excel() == []:
                self.cal()
                self.lg = False
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

        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])
        (xe, ye, out)= (xt, rlg, out)
        if self.lg==True:

            graph_title = self.nlog.text()
            graph_x = self.xlog.text()
            graph_y = self.ylog.text()

            plt.scatter(x, y, color=(self.logarojo(),self.logaverde(),self.logaazul()), label='Datos experimentales')

            plt.plot(xe, ye,color=(self.glogrojo(),self.glogverde(),self.glogazul()), label='Regresion encontrada')
            plt.suptitle(graph_title)
            plt.xlabel(graph_x)
            plt.ylabel(graph_y)
            plt.grid(b=True)
            plt.legend()
            if self.explg==True:
                plt.savefig("graficas/"+graph_title)
                plt.close()
                self.gu()
            self.explg = False
            pyplot.axhline(0, color="black")
            pyplot.axvline(0, color="black")
            if self.grlg==True:
                self.reglg.setText(out)
                plt.show()
                plt.close()
            self.grlg=False

        self.lg=False
        yp= a*(x)**b
        
        
        R2 = (np.var(yp))/(np.var(y))
        
        
        
        return R2
        
    def regresionslg(self):
        if self.pos==False:
            if self.cantidad() == None:
                self.erdatos()
                self.sg = False
                return None
            if self.datos() == None:
                self.sg = False
                return None
            datos = np.array(self.datos())
        if self.pos==True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                self.sg = False
                return None
            if self.excel() == []:
                self.cal()
                self.sg = False
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

        x = np.array(datos[:, 0])
        y = np.array(datos[:, 1])
        (xe, ye, out) = (xt, rslg, out)

        if self.sg==True:

            graph_title = self.nsemi.text()
            graph_x = self.xsemi.text()
            graph_y = self.ysemi.text()

            plt.scatter(x, y, color=(self.psemirojo(),self.psemiverde(),self.psemiazul()), label='Datos experimentales')
            plt.plot(xe, ye,color=(self.gsemirojo(),self.gsemiverde(),self.gsemiazul()), label='Regresion encontrada')

            plt.suptitle(graph_title)
            plt.xlabel(graph_x)
            plt.ylabel(graph_y)
            plt.grid(b=True)
            plt.legend()
            if self.expsg==True:
                plt.savefig("graficas/"+graph_title)
                plt.close()
                self.gu()
            self.expsg = False
            pyplot.axhline(0, color="black")
            pyplot.axvline(0, color="black")
            if self.grsg==True:
                self.regslg.setText(out)
                plt.show()
                plt.close()
            self.grsg=False
        self.sg=False
        
        yp = b*(e)**(m*x)
        
        R2 = (np.var(yp))/(np.var(y))
        return R2
    def regresionpol(self):
        if self.pos==False:
            if self.cantidad() == None:
                self.erdatos()
                self.pl = False
                return None
            if self.datos() == None:
                self.pl = False
                return None
            datos = np.array(self.datos())
        if self.pos==True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                self.pl = False
                return None
            if self.excel() == []:
                self.cal()
                self.pl = False
                return None
            datos=np.array(self.excel())

        if self.grd==False:
            try:
                g = int(self.grad.text())


            except:
                self.gran()
                self.pl = False
                return None
        if self.grd==True:
            try:
                g = int(self.gra.text())


            except:
                self.gran()
                self.pl = False
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

        (xe, ye, out)=(xt, p, out)

        if self.pl==True:

            graph_title = self.npol.text()
            graph_x = self.xpol.text()
            graph_y = self.ypol.text()

            plt.scatter(x, y, color=(self.ppolrojo(),self.ppolverde(),self.ppolazul()), label='Datos experimentales')
            plt.plot(xe, ye,color=(self.gpolrojo(),self.gpolverde(),self.gpolazul()), label='Regresion encontrada')

            plt.suptitle(graph_title)
            plt.xlabel(graph_x)
            plt.ylabel(graph_y)
            plt.grid(b=True)
            plt.legend()
            if self.exppl==True:
                plt.savefig("graficas/"+graph_title)
                plt.close()
                self.gu()
            self.exppl = False
            pyplot.axhline(0, color="black")
            pyplot.axvline(0, color="black")
            if self.grpol==True:
                self.cua.setText(out)
                plt.show()
                plt.close()
            self.grpol=False

        self.pl=False
        
        yp= PolyCoefficients(x,sol)
        R2 = (np.var(yp))/(np.var(y))
        
        return R2

    def sinusoidal(self):
        
        g = int(20)
        if self.pos == False:
            if self.cantidad() == None:
                self.erdatos()
                self.sn = False
                return None
            if self.datos() == None:
                self.sn = False
                return None
            datos = np.array(self.datos())
        if self.pos == True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                self.sn = False
                return None
            if self.excel() == []:
                self.cal()
                self.sn = False
                return None
            datos = np.array(self.excel())
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
        
        def fourier (w): #Ajuste datos

            n = np.size(datos[:, 0])
            A = np.empty([3,3])
            A[0,0] = n
            A[0,1] = np.sum(np.cos(w*x))
            A[1,0] =A[0,1]
            A[0,2] = np.sum(np.sin(w*x))
            A[2,0] = A[0,2]
            A[1,1] = np.sum(np.cos(w*x)**2)
            A[2,2] = np.sum(np.sin(w*x)**2)
            A[1,2] = np.sum(np.sin(w*x)*np.cos(w*x))
            A[2,1] = A[1,2]
            
            B= np.empty((3))
            B[0] = np.sum(y)
            B[1] = np.sum(y*np.cos(w*x))
            B[2] = np.sum(y*np.sin(w*x)) 
                
            sol = np.linalg.solve(A,B)

            return w , sol
        
        def coef_reg(w , sol):

            (A0, A1, B1) = sol
            
            yp = A0 + A1*np.cos(w*x) + B1*np.sin(w*x)
            
            R2 = (np.var(yp)) / (np.var(y))

            return R2
        
        def optimize(w1):
            wi = w1 - 0.5
            wmin = []
            try:
                for l in np.linspace(wi , wi + 1.001 , 1000):
                    coe= 1 - coef_reg(fourier(l)[0] , fourier(l)[1] )
                    wmin.append(coe)
            except:
                None

            for k in range(len(wmin)):
                if wmin[k] == np.min(wmin):
                    break
            wfinal = wi + (k/1000)
            (wfinal , tup ) = fourier(wfinal)
            A0 = tup[0]
            A1 = tup[1]
            B1 = tup[2]
            return wfinal , A0 , A1 ,B1
            
        
        
        if k > 0:  # Seno

            w_aprox = np.sqrt((abs(solu[3]) * 6) / (solu[1]))

            w , c , A, B = optimize(w_aprox)
            out = '{}sin({}x)+{}'.format(B, w, c)

            seno = B * np.sin(w * xt) + c

            if self.sn == True:

                plt.plot(x, y)

                graph_title = self.nsin.text()
                graph_x = self.xsin.text()
                graph_y = self.ysin.text()

                plt.scatter(x, y, color=(self.senrojo(), self.senverde(), self.senazul()), label='Datos experimentales')
                plt.plot(xt, seno, color=(self.gsenrojo(), self.gsenverde(), self.gsenazul()), label=out)
                pyplot.axhline(0, color="black")
                pyplot.axvline(0, color="black")
                plt.suptitle(graph_title)
                plt.xlabel(graph_x)
                plt.ylabel(graph_y)
                plt.grid(b=True)
                plt.legend()
                if self.expsn == True:
                    plt.savefig("graficas/" + graph_title)
                    plt.close()
                    self.gu()
                self.expsn = False
                if self.grsn == True:
                    self.sin.setText(out)
                    plt.show()
                    plt.close()
                self.grsn = False

            self.sn = False
            yp = A * np.sin(w * x) + c

            R2 = (np.var(yp)) / (np.var(y))

            return R2

        else:  # Coseno

            w_aprox = np.sqrt((abs(solu[4]) * 24) / (2 * abs(solu[2])))

            w , c , A, B = optimize(w_aprox)


            out = '{}cos({}x)+{}'.format(A, w, c)

            coseno = A * np.cos(w * xt) + c

            if self.sn == True:
                plt.plot(x, y)
                graph_title = self.nsin.text()
                graph_x = self.xsin.text()
                graph_y = self.ysin.text()

                plt.scatter(x, y, color=(self.senrojo(), self.senverde(), self.senazul()), label='Datos experimentales')
                plt.plot(xt, coseno, color=(self.gsenrojo(), self.gsenverde(), self.gsenazul()), label=out)
                pyplot.axhline(0, color="black")
                pyplot.axvline(0, color="black")
                plt.suptitle(graph_title)
                plt.xlabel(graph_x)
                plt.ylabel(graph_y)
                plt.grid(b=True)
                plt.legend()
                if self.expsn == True:
                    plt.savefig("graficas/" + graph_title)
                    plt.close()
                    self.gu()
                self.expsn = False
                if self.grsn == True:
                    self.sin.setText(out)
                    plt.show()
                    plt.close()
                self.grsn = False


            yp = A * np.cos(w * x) + c

            R2 = (np.var(yp)) / (np.var(y))

            return R2
    def re_auto(self):
        if self.pos==False:
            if self.cantidad() == None:
                self.erdatos()
                return None
            if self.datos() == None:
                return None

        if self.pos==True:
            if self.excel() == None or self.excel() == 0 or self.excel() == 1:
                self.erdatos()
                self.pos=False

                return None
            if self.excel() == []:
                self.cal()
                return None
        self.grd = True
        if self.regresionpol()==None:
            return None


        #Se llaman las regresiones para almacenar los valores en una tupla llamada tp
        (rlineal,rlog,rsemilog,rpoli,rsinu) = self.reglineal(),self.regresionlg(),self.regresionslg(),self.regresionpol(),self.sinusoidal()

        tup = (rlineal,rlog,rsemilog,rpoli,rsinu)
        self.grd=False
        k= []
        ##Se añade a la lista k la distancia de r a 1, y se almacenan en el mismo orden que tup
        for i in tup:
            r=np.abs(i-1)
            k.append(r)

        ##Se encuentra el índice del valor mínimo de tup para así saber a que regresión hace referencia
        for p in range(len(k)):
            if k[p] == np.min(k):
                break

        ##Esta tupla tiene el mismo orden de tup y sencillamente sirve para devolver cúal sería la regresion adecuada
        tupstr= ('Regresión lineal','Regresión logaritmica','Regresión semi-logaritmica','Regresión polinomial','Regresión sinusoidal')
        self.atreg.setText("La mejor regresión es: ")
        self.autr.setText(tupstr[p])
    #Modelos de probabilidad
    def bernoulli(self):
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
        ax.scatter(x, fmp, color=(self.bernrojo(),self.bernverde(),self.bernazul()))
        ax.vlines(x, 0, fmp, colors=(self.bernrojo(),self.bernverde(),self.bernazul()), lw=5, alpha=0.5)
        ax.set_yticks([0., 0.2, 0.4, 0.6])
        title=self.nberno.text()


        plt.title(title)


        plt.ylabel(self.yberno.text())
        plt.xlabel(self.xberno.text())
        if self.expbr==True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.expbr=False
        if self.grabn==True:
            plt.show()
            plt.close()
            self.grabn=False
    def grabern(self):
        self.grabn=True

    def expber(self):
        self.expbr=True
        self.bernoulli()

    def histbernoulli(self):
        np.random.seed(2016)  # replicar random
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
        plt.hist(aleatorios,color=(self.hbernrojo(),self.hbernverde(),self.hbernazul()))
        title=self.hberno.text()

        plt.ylabel(self.hyberno.text())
        plt.xlabel(self.hxberno.text())
        plt.title(title)
        if self.hexpbr==True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.hexpbr = False
        if self.hgrabn==True:

            plt.show()
            plt.close()
            self.hgrabn=False
    def hgrabrn(self):
        self.hgrabn = True
    def hexpbern(self):
        self.hexpbr = True
        self.histbernoulli()
    def binomial(self):
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
        plt.vlines(x, 0, fmp, colors=(self.binorojo(),self.binoverde(),self.binoazul()), lw=5, alpha=0.5)
        title=self.nbin.text()
        plt.title(title)
        if self.expbi==True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.expbi = False
        plt.ylabel(self.ybin.text())
        plt.xlabel(self.xbin.text())
        if self.grabi==True:

            plt.show()
            plt.close()
            self.grabi=False
    def grabin(self):
        self.grabi = True
    def exporbin(self):
        self.expbi = True
        self.binomial()
    def histbinomial(self):

        np.random.seed(2016)  # replicar random
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
        plt.hist(aleatorios, color=(self.hbinorojo(),self.hbinoverde(),self.hbinoazul()))
        title=self.hbin.text()

        if self.exphbi == True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.exphbi = False
        plt.ylabel(self.hybin.text())
        plt.xlabel(self.hxbin.text())
        plt.title(title)
        if self.hgrabi==True:

            plt.show()
            plt.close()
            self.hgrabi=False
    def grahbin(self):
        self.hgrabi =True
    def exphbin(self):
        self.exphbi = True
        self.histbinomial()
    def exponencial(self):
        try:
            a = float(self.para.text())
        except:
            self.err()
            return None
        exponencial = stats.expon(a)
        x = np.linspace(exponencial.ppf(0.01),
                        exponencial.ppf(0.99), 100)
        fp = exponencial.pdf(x)  # Función de Probabilidad
        plt.plot(x,fp ,color=(self.exprojo(),self.expverde(),self.expazul()))
        title=self.nexpon.text()
        plt.title(title)
        if self.exppo == True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.exppo = False
        plt.ylabel(self.yexpon.text())
        plt.xlabel(self.xexpon.text())
        if self.grapo == True:
            plt.show()
            plt.close()
        self.grapo = False
    def grafexp(self):
        self.grapo = True
    def exporexp(self):
        self.exppo = True
        self.exponencial()
    def histexponencial(self):

        np.random.seed(2016)  # replicar random
        try:
            a = float(self.para.text())
        except:
            self.err()
            return None
        exponencial = stats.expon(a)
        aleatorios = exponencial.rvs(1000)  # genera aleatorios
        plt.hist(aleatorios, color=(self.hexprojo(),self.hexpverde(),self.hexpazul()))
        title=self.hexpon.text()
        if self.exphpo == True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.exphpo = False
        plt.ylabel(self.hyexpon.text())
        plt.xlabel(self.hxexpon.text())
        plt.title(title)
        if self.hgrapo == True:
            plt.show()
            plt.close()
        self.hgrapo=False
    def hgrafexp(self):
        self.hgrapo = True
    def hexporexp(self):
        self.exphpo = True
        self.histexponencial()
    def hipergeometrica(self):
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
        plt.vlines(x, 0, fmp, colors=(self.hiperrojo(),self.hiperverde(),self.hiperazul()), lw=5, alpha=0.5)
        title=self.nhip.text()
        plt.title(title)
        if self.exphi == True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.exphi = False
        plt.ylabel(self.yhip.text())
        plt.xlabel(self.xhip.text())
        if self.grahi == True:
            plt.show()
            plt.close()
            self.grahi =False
    def grafhip(self):
        self.grahi = True
    def exporhip(self):
        self.exphi = True
        self.hipergeometrica()
    def histhipergeometrica(self):

        np.random.seed(2016)  # replicar random
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
        plt.hist(aleatorios, color=(self.hhiperrojo(),self.hhiperverde(),self.hhiperazul()))
        title=self.hhip.text()
        if self.exphhi == True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.exphhi = False
        plt.ylabel(self.hyhip.text())
        plt.xlabel(self.hxhip.text())
        plt.title(title)
        if self.hgrahi == True:
            plt.show()
            plt.close()
            self.hgrahi = False
    def hghip(self):
        self.hgrahi = True
    def hexporhip(self):
        self.exphhi = True
        self.histhipergeometrica()
    def normal(self):
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
        plt.plot(x,fp ,color=(self.normalrojo(),self.normalverde(),self.normalazul()))
        title=self.nnorm.text()
        plt.title(title)
        if self.expno == True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.expno = False
        plt.ylabel(self.ynorm.text())
        plt.xlabel(self.xnorm.text())
        if self.grano == True:
            plt.show()
            plt.close()
        self.grano = False
    def grafno(self):
        self.grano = True
    def exprnon(self):
        self.expno = True
        self.normal()
    def histnormal(self):
        np.random.seed(2016)  # replicar random
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
        plt.hist(aleatorios, color=(self.hnormalrojo(),self.hnormalverde(),self.hnormalazul()))
        title=self.hnorm.text()
        if self.exphno == True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.exphno = False
        plt.ylabel(self.hynorm.text())
        plt.xlabel(self.hxnorm.text())
        plt.title(title)
        if self.hgrano == True:
            plt.show()
            plt.close()
            self.hgrano = False
    def histgrano(self):
        self.hgrano = True
    def histexpno(self):
        self.exphno = True
        self.histnormal()
    def poisson(self):
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
        plt.vlines(x, 0, fmp, colors=(self.poisrojo(),self.poisverde(),self.poisazul()), lw=5, alpha=0.5)
        title=self.npoi.text()
        plt.title(title)
        if self.exppoi == True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.exppoi = False
        plt.ylabel(self.ypoi.text())
        plt.xlabel(self.xpoi.text())
        if self.grapoi == True:
            plt.show()
            plt.close()
            self.grapoi = False
    def grafpoi(self):
        self.grapoi = True
    def expopoi(self):
        self.exppoi = True
        self.poisson()
    def histpoisson(self):

        np.random.seed(2016)  # replicar random

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
        plt.hist(aleatorios, color=(self.hpoisrojo(),self.hpoisverde(),self.hpoisazul()))
        title=self.hpoi.text()
        if self.exphpo == True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
            self.exphpo = False
        plt.ylabel(self.hypoi.text())
        plt.xlabel(self.hxpoi.text())
        plt.title(title)
        if self.hgrapo == True:
            plt.show()
            plt.close()
            self.hgrapo = False
    def grafhpo(self):
        self.hgrapo = True
    def exporhpo(self):
        self.exphpo = True
        self.histpoisson()
    def uniforme(self):

        try:
            m = float(self.val.text())
        except:
            self.err()
            return None
        try:
            v=float(self.longi.text())
        except:
            self.err()
            return None
        if v<=0:
            self.neguni()
            return None
        uniforme = stats.uniform(m, v)
        x = np.linspace(uniforme.ppf(0.01),
                        uniforme.ppf(0.99), 100)
        fp = uniforme.pdf(x)  # Función de Probabilidad
        fig, ax = plt.subplots()
        ax.plot(x, fp, '--')
        ax.vlines(x, 0, fp, colors=(self.unifrojo(), self.unifverde(), self.unifazul()), lw=5, alpha=0.5)
        ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2])
        title=self.nuni.text()
        if self.expun == True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
        self.expun = False
        plt.title(title)
        plt.ylabel(self.yuni.text())
        plt.xlabel(self.xuni.text())
        if self.graun== True:
            plt.show()
            plt.close()
        self.graun = False
    def grafun(self):
        self.graun = True

    def exporun(self):
        self.expun = True
        self.uniforme()
    def huniforme(self):
        try:
            m = float(self.val.text())
        except:
            self.err()
            return None
        try:
            v=float(self.longi.text())
        except:
            self.err()
            return None
        if v<=0:
            self.neguni()
            return None
        uniforme = stats.uniform(m, v)
        aleatorios = uniforme.rvs(1000)  # genera aleatorios
        plt.hist(aleatorios, 20, color=(self.hiunifrojo(), self.hiunifverde(), self.hiunifazul()))
        title=self.nhuni.text()
        if self.exphun == True:
            plt.savefig("graficas/" + title)
            plt.close()
            self.gu()
        self.exphun = False
        plt.ylabel(self.yhuni.text())
        plt.xlabel(self.xhuni.text())
        plt.title(title)
        if self.hgraun == True:
            plt.show()
            plt.close()
        self.hgraun = False
    def grafhun(self):
        self.hgraun = True
        self.huniforme()
    def exporhun(self):
        self.exphun = True
        self.huniforme()
    #Probabilidades
    def simple(self):
        try:
            Totales = int(self.sct.text())
            Satisfactorios = int(self.scf.text())
        except:
            self.err()
            return None

        if Totales==0:
            Probabilidad=0
        else:
            Probabilidad = Satisfactorios / Totales
        ProbabilidadPorcentual = Probabilidad * 100
        self.sim.setText('La Probabilidad es: '+ str(round(ProbabilidadPorcentual, 4)) + '%')
    def permutacion(self):
        try:
            n = int(self.pnum.text())
            k = int(self.pcan.text())
        except:
            self.err()
            return None
        if k>n:
            self.casos()
            return None
        Permutaciones = math.factorial(n) / math.factorial(n - k)
        self.pet.setText('De '+ str(n)+ ' casos se obtiene un arreglo de: '+str(Permutaciones))
    def combinaciones(self):
        try:
            n = int(self.cm.text())
            k = int(self.cel.text())
        except:
            self.err()
            return None
        if k>n:
            self.casos()
            return None
        Permutaciones = math.factorial(n) / math.factorial(n - k)
        Combinaciones = Permutaciones / math.factorial(k)
        self.comn.setText('De '+ str(n)+ ' casos se obtiene un arreglo de: '+ str(Combinaciones))
    def valor(self):
        try:
            Valor = float(self.valo.text())
            Probabilidad = float(self.vpro.text())
        except:
            self.err()
            return None
        ValorEsperado = Valor * Probabilidad
        self.vales.setText('El valor esperado es: '+ str(ValorEsperado))
    def pbinomial(self):
        try:
            n = int(self.dct.text())
            k = int(self.dcs.text())
            p = float(self.dps.text())
        except:
            self.err()
            return None
        if k>n:
            self.tot()
            return None
        n_k = math.factorial(n) / (math.factorial(n - k) * math.factorial(k))
        pk = p ** k
        pk2 = (1 - p) ** (n - k)
        y = n_k * pk * pk2
        self.yyy.setText("exactamente "+str(k)+ " éxitos en "+ str(n)+" pruebas si la probabilidad de")
        self.yy.setText("éxito en una sola prueba es " +str(p)+ ' es igual a: '+str(y))
        media = n * p
        self.med.setText('La Media es: '+str(media))
        varianza = media * (1 - p)
        self.vari.setText('La Varianza es: '+str(varianza))
    def pgeometrica(self):
        try:
            r = int(self.gnp.text())
            p = float(self.gps.text())
        except:
            self.err()
            return None
        k = (1 - p) ** (r - 1)
        y = p * k
        self.dgeo.setText('La probabilidad del exito en '+str(r)+' intentos es: '+str(y))
        if p==0:
            media=0
        else:
            media = 1 / p
        self.medgeo.setText('La media es: '+str(media))
        if p==0:
            varianza=0
        else:
            varianza = (1 - p) / (p ** 2)
        self.vargeo.setText("La varianza es: "+str(varianza))
    def phiper(self):
        try:
            N = int(self.het.text())
            d = int(self.hec.text())
            n = int(self.hest.text())
            x = int(self.hde.text())
        except:
            self.err()
            return None
        a = N - d
        b = n - x
        c = math.factorial(d) / (math.factorial(d - x) * math.factorial(x))
        d = math.factorial(a) / (math.factorial(a - b) * math.factorial(b))
        e = math.factorial(N) / (math.factorial(N - n) * math.factorial(n))
        y = (c * d) / e
        self.prhip.setText('La Probabilidad de encontrar '+str(x)+' objetos en el grupo es:'+str(y))
        media = (n * d) / N
        self.medhip.setText('La media es: '+str(media))
        varianza = (n * (d / N) * (1 - (d / N)) * (N - n)) / (N - 1)
        self.varhip.setText('La varianza es: '+str(varianza))
    def ppoisson(self):
        try:
            x = float(self.prm.text())
            a = float(self.pit.text())
            k = float(self.pnus.text())
        except:
            self.err()
            return None
        alfa = x * a
        c = math.exp(-alfa)
        d = alfa ** k
        y = (c * d) / math.factorial(k)
        self.prpoi.setText("La Probabilidad de que ocurra "+str(k)+ " sucesos ")
        self.prpois.setText("en el tiempo "+str(a)+" es:"+str(y))
        media = alfa
        self.medpoi.setText('La media es: '+str(media))
        varianza = alfa
        self.varpoi.setText('La varianza es: '+str(varianza))
    def pexponencial(self):
        try:
            x = float(self.expx.text())
            a = float(self.expa.text())
        except:
            self.err()
            return None
        exponencial = stats.expon(a)
        p = exponencial.pdf(x)
        self.proex.setText('la probabilidad del suceso X es: '+str(p))
    def pnormal(self):
        try:
            x = float(self.nix.text())
            u = float(self.niu.text())
            o = float(self.nip.text())
        except:
            self.err()
            return None
        if o==0:
            z=0
        else:
            z = (x - u) / o
        p = stats.norm.cdf(z)
        self.prexpo.setText('la probabilidad del suceso X es menor o igual a:'+str(p))
    #colores regresiones
    def rojo(self):
        r = self.srojo.value()
        self.lrojo.setText(str(r))
        r=r/100
        return r
    def azul(self):
        a = self.sazul.value()
        self.lazul.setText(str(a))
        a=a/100
        return a
    def verde(self):
        v = self.sverde.value()
        self.lverde.setText(str(v))
        v=v/100
        return v
    def punrojo(self):
        r = self.rojop.value()
        self.projo.setText(str(r))
        r=r/100
        return r
    def punverde(self):
        v = self.verdep.value()
        self.pverde.setText(str(v))
        v=v/100
        return v
    def punazul(self):
        a = self.azulp.value()
        self.pazul.setText(str(a))
        a=a/100
        return a
    def linealrojo(self):
        r = self.linrojo.value()
        self.rojolin.setText(str(r))
        r=r/100
        return r
    def linealverde(self):
        v = self.linverde.value()
        self.verdelin.setText(str(v))
        v=v/100
        return v
    def linealazul(self):
        a = self.linazul.value()
        self.azullin.setText(str(a))
        a=a/100
        return a
    def logarojo(self):
        r = self.logrojo.value()
        self.rojolog.setText(str(r))
        r=r/100
        return r
    def logaverde(self):
        v = self.logverde.value()
        self.verdelog.setText(str(v))
        v=v/100
        return v
    def logaazul(self):
        a = self.logazul.value()
        self.azullog.setText(str(a))
        a=a/100
        return a
    def glogrojo(self):
        r = self.lorojo.value()
        self.rojolo.setText(str(r))
        r=r/100
        return r
    def glogverde(self):
        v = self.loverde.value()
        self.verdelo.setText(str(v))
        v=v/100
        return v
    def glogazul(self):
        a = self.loazul.value()
        self.azullo.setText(str(a))
        a=a/100
        return a
    def psemirojo(self):
        r = self.semirojo.value()
        self.rojosemi.setText(str(r))
        r=r/100
        return r
    def psemiverde(self):
        v = self.semiverde.value()
        self.verdesemi.setText(str(v))
        v=v/100
        return v
    def psemiazul(self):
        a = self.semiazul.value()
        self.azulsemi.setText(str(a))
        a=a/100
        return a
    def gsemirojo(self):
        r = self.semrojo.value()
        self.rojosem.setText(str(r))
        r=r/100
        return r
    def gsemiverde(self):
        v = self.semverde.value()
        self.verdesem.setText(str(v))
        v=v/100
        return v
    def gsemiazul(self):
        a = self.semazul.value()
        self.azulsem.setText(str(a))
        a=a/100
        return a
    def ppolrojo(self):
        r = self.polrojo.value()
        self.rojopol.setText(str(r))
        r=r/100
        return r
    def ppolverde(self):
        v = self.polverde.value()
        self.verdepol.setText(str(v))
        v=v/100
        return v
    def ppolazul(self):
        a = self.polazul.value()
        self.azulpol.setText(str(a))
        a=a/100
        return a
    def gpolrojo(self):
        r = self.porojo.value()
        self.rojopo.setText(str(r))
        r=r/100
        return r
    def gpolverde(self):
        v = self.poverde.value()
        self.verdepo.setText(str(v))
        v=v/100
        return v
    def gpolazul(self):
        a = self.poazul.value()
        self.azulpo.setText(str(a))
        a=a/100
        return a
    #colores distribuciones e histogramas
    def bernrojo(self):
        r = self.berrojo.value()
        self.rojober.setText(str(r))
        r=r/100
        return r
    def bernverde(self):
        v = self.berverde.value()
        self.verdeber.setText(str(v))
        v=v/100
        return v
    def bernazul(self):
        a = self.berazul.value()
        self.azulber.setText(str(a))
        a=a/100
        return a
    def hbernrojo(self):
        r = self.hbrojo.value()
        self.rojohb.setText(str(r))
        r=r/100
        return r
    def hbernverde(self):
        v = self.hbverde.value()
        self.verdehb.setText(str(v))
        v=v/100
        return v
    def hbernazul(self):
        a = self.hbazul.value()
        self.azulhb.setText(str(a))
        a=a/100
        return a
    def binorojo(self):
        r = self.binrojo.value()
        self.rojobin.setText(str(r))
        r=r/100
        return r
    def binoverde(self):
        v = self.binverde.value()
        self.verdebin.setText(str(v))
        v=v/100
        return v
    def binoazul(self):
        a = self.binazul.value()
        self.azulbin.setText(str(a))
        a=a/100
        return a
    def hbinorojo(self):
        r = self.hbirojo.value()
        self.rojohbi.setText(str(r))
        r=r/100
        return r
    def hbinoverde(self):
        v = self.hbiverde.value()
        self.verdehbi.setText(str(v))
        v=v/100
        return v
    def hbinoazul(self):
        a = self.hbiazul.value()
        self.azulhbi.setText(str(a))
        a=a/100
        return a
    def exprojo(self):
        r = self.exrojo.value()
        self.rojoex.setText(str(r))
        r=r/100
        return r
    def expverde(self):
        v = self.exverde.value()
        self.verdeex.setText(str(v))
        v=v/100
        return v
    def expazul(self):
        a = self.exazul.value()
        self.azulex.setText(str(a))
        a=a/100
        return a
    def hexprojo(self):
        r = self.herojo.value()
        self.rojohe.setText(str(r))
        r=r/100
        return r
    def hexpverde(self):
        v = self.heverde.value()
        self.verdehe.setText(str(v))
        v=v/100
        return v
    def hexpazul(self):
        a = self.heazul.value()
        self.azulhe.setText(str(a))
        a=a/100
        return a
    def normalrojo(self):
        r = self.norrojo.value()
        self.rojonor.setText(str(r))
        r=r/100
        return r
    def normalverde(self):
        v = self.norverde.value()
        self.verdenor.setText(str(v))
        v=v/100
        return v
    def normalazul(self):
        a = self.norazul.value()
        self.azulnor.setText(str(a))
        a=a/100
        return a
    def hnormalrojo(self):
        r = self.hnrojo.value()
        self.rojohn.setText(str(r))
        r=r/100
        return r
    def hnormalverde(self):
        v = self.hnverde.value()
        self.verdehn.setText(str(v))
        v=v/100
        return v
    def hnormalazul(self):
        a = self.hnazul.value()
        self.azulhn.setText(str(a))
        a=a/100
        return a
    def hiperrojo(self):
        r = self.hiprojo.value()
        self.rojohip.setText(str(r))
        r=r/100
        return r
    def hiperverde(self):
        v = self.hipverde.value()
        self.verdehip.setText(str(v))
        v=v/100
        return v
    def hiperazul(self):
        a = self.hipazul.value()
        self.azulhip.setText(str(a))
        a=a/100
        return a
    def hhiperrojo(self):
        r = self.hhrojo.value()
        self.rojohh.setText(str(r))
        r=r/100
        return r
    def hhiperverde(self):
        v = self.hhverde.value()
        self.verdehh.setText(str(v))
        v=v/100
        return v
    def hhiperazul(self):
        a = self.hhazul.value()
        self.azulhh.setText(str(a))
        a=a/100
        return a
    def poisrojo(self):
        r = self.poirojo.value()
        self.rojopoi.setText(str(r))
        r=r/100
        return r
    def poisverde(self):
        v = self.poiverde.value()
        self.verdepoi.setText(str(v))
        v=v/100
        return v
    def poisazul(self):
        a = self.poiazul.value()
        self.azulpoi.setText(str(a))
        a=a/100
        return a
    def hpoisrojo(self):
        r = self.hprojo.value()
        self.rojohp.setText(str(r))
        r=r/100
        return r
    def hpoisverde(self):
        v = self.hpverde.value()
        self.verdehp.setText(str(v))
        v=v/100
        return v
    def hpoisazul(self):
        a = self.hpazul.value()
        self.azulhp.setText(str(a))
        a=a/100
        return a
    def senrojo(self):
        r = self.sirojo.value()
        self.rojosi.setText(str(r))
        r=r/100
        return r
    def senverde(self):
        v = self.siverde.value()
        self.verdesi.setText(str(v))
        v=v/100
        return v
    def senazul(self):
        a = self.siazul.value()
        self.azulsi.setText(str(a))
        a=a/100
        return a
    def gsenrojo(self):
        r = self.serojo.value()
        self.rojose.setText(str(r))
        r=r/100
        return r
    def gsenverde(self):
        v = self.severde.value()
        self.verdese.setText(str(v))
        v=v/100
        return v
    def gsenazul(self):
        a = self.seazul.value()
        self.azulse.setText(str(a))
        a=a/100
        return a
    def unifrojo(self):
        r = self.unrojo.value()
        self.rojoun.setText(str(r))
        r=r/100
        return r
    def unifverde(self):
        v = self.unverde.value()
        self.verdeun.setText(str(v))
        v=v/100
        return v
    def unifazul(self):
        a = self.unazul.value()
        self.azulun.setText(str(a))
        a=a/100
        return a
    def hiunifrojo(self):
        r = self.hunrojo.value()
        self.rojohun.setText(str(r))
        r=r/100
        return r
    def hiunifverde(self):
        v = self.hunverde.value()
        self.verdehun.setText(str(v))
        v=v/100
        return v
    def hiunifazul(self):
        a = self.hunazul.value()
        self.azulhun.setText(str(a))
        a=a/100
        return a

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
    def neguni(self):
        msg = QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("La longitud debe ser mayor a cero")
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
    def gran(self):
        msg = QMessageBox()
        msg.setWindowTitle("ERRO")
        msg.setText("No ha ingresado un valor valido para los grados de su polinomio")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def gu(self):
        msg = QMessageBox()
        msg.setWindowTitle("Gráfica guardada")
        msg.setText("Se ha guardado la gráfica")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()
    def casos(self):
        msg = QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("Los elementos elegidos deben ser menores o iguales al total de elementos")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    def tot(self):
        msg = QMessageBox()
        msg.setWindowTitle("ERROR")
        msg.setText("Los casos favorables deben ser menores o iguales al total de los casos")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
if __name__ == "__main__":

    prog = QApplication(sys.argv)
    GUI = programa()
    GUI.show()
    sys.exit(prog.exec_())