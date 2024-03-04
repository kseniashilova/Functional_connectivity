from matplotlib import pyplot as plt

import new_main
from network import ER_network, WS_network
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QDoubleSpinBox, QSpinBox, QHBoxLayout, \
    QLabel, QSlider, QPushButton, QCheckBox, QDialog
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    figure = Figure()
    ax = figure.add_subplot(111)

    def initUI(self):
        self.setGeometry(300, 300, 1000, 700)

        layout = QVBoxLayout()

        # initial values
        self.networktype = None
        self.lowerBound = None
        self.upperBound = None
        self.prob = None
        self.methods_list = None
        self.distrib = None
        self.whitenoise=False


        # Choose network type
        self.comboBoxNetworkType = QComboBox(self)
        self.comboBoxNetworkType.addItem("ER")
        self.comboBoxNetworkType.addItem("WS")
        # self.comboBoxNetworkType.setCurrentIndex(-1)
        self.comboBoxNetworkType.activated[str].connect(self.updateNetworkType)
        layout.addWidget(self.comboBoxNetworkType)


        # Choose range of nodes
        hbox = QHBoxLayout()

        self.lowerSliderLabel = QLabel("Lower Boundary", self)
        hbox.addWidget(self.lowerSliderLabel)
        self.lowerSlider = QSlider(Qt.Horizontal, self)
        self.lowerSlider.setMinimum(0)
        self.lowerSlider.setMaximum(100)
        self.lowerSlider.setValue(5)
        self.lowerSlider.valueChanged.connect(self.updateLowerBound)
        hbox.addWidget(self.lowerSlider)

        self.upperSliderLabel = QLabel("Upper Boundary", self)
        hbox.addWidget(self.upperSliderLabel)
        self.upperSlider = QSlider(Qt.Horizontal, self)
        self.upperSlider.setMinimum(0)
        self.upperSlider.setMaximum(100)
        self.upperSlider.setValue(20)
        self.upperSlider.valueChanged.connect(self.updateUpperBound)
        hbox.addWidget(self.upperSlider)

        layout.addLayout(hbox)


        # Choose probability for network
        hbox_prob = QHBoxLayout()

        self.probSliderLabel = QLabel("Conn prob", self)
        hbox_prob.addWidget(self.probSliderLabel)
        self.probSlider = QSlider(Qt.Horizontal, self)
        self.probSlider.setMinimum(0)
        self.probSlider.setMaximum(100)
        self.probSlider.setValue(15)
        self.probSlider.valueChanged.connect(self.updateProb)
        hbox_prob.addWidget(self.probSlider)
        layout.addLayout(hbox_prob)


        # Choose weights coefficient
        hbox_weights = QHBoxLayout()

        self.weightsSliderLabel = QLabel("weights=coeff/N", self)
        hbox_weights.addWidget(self.weightsSliderLabel)
        self.weightsSlider = QSlider(Qt.Horizontal, self)
        self.weightsSlider.setMinimum(0)
        self.weightsSlider.setMaximum(7000)
        self.weightsSlider.setValue(1000)
        self.weightsSlider.valueChanged.connect(self.updateWeightCoef)
        hbox_weights.addWidget(self.weightsSlider)

        layout.addLayout(hbox_weights)

        # Choose distrib weights
        self.comboBoxDistribType = QComboBox(self)
        self.comboBoxDistribType.addItem("Gaussian")
        self.comboBoxDistribType.addItem("Uniform")
        # self.comboBoxNetworkType.setCurrentIndex(-1)
        self.comboBoxDistribType.activated[str].connect(self.updateDistrib)
        layout.addWidget(self.comboBoxDistribType)

        # white noise checkbox
        self.checkbox_noise = QCheckBox('White noise to inputs', self)
        self.checkbox_noise.stateChanged.connect(self.checkboxWhiteNoiseChanged)
        layout.addWidget(self.checkbox_noise)

        # Create checkboxes
        self.checkbox1 = QCheckBox('Pearson Correlation', self)
        self.checkbox1.stateChanged.connect(self.checkboxStateChanged)
        layout.addWidget(self.checkbox1)

        self.checkbox2 = QCheckBox('Transfer Entropy', self)
        self.checkbox2.stateChanged.connect(self.checkboxStateChanged)
        layout.addWidget(self.checkbox2)

        self.checkbox3 = QCheckBox('CCG', self)
        self.checkbox3.stateChanged.connect(self.checkboxStateChanged)
        layout.addWidget(self.checkbox3)

        self.checkbox4 = QCheckBox('GLM', self)
        self.checkbox4.stateChanged.connect(self.checkboxStateChanged)
        layout.addWidget(self.checkbox4)



        # Plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)


        # launch button
        self.launchButton = QPushButton('Start sim', self)
        self.launchButton.clicked.connect(self.launch)
        layout.addWidget(self.launchButton)


        self.setLayout(layout)
        self.show()

    def updateLowerBound(self, value):
        if value >= self.upperSlider.value():
            self.lowerSlider.setValue(self.upperSlider.value() - 1)
        self.lowerBound = value
        self.lowerSliderLabel.setText(f"Lower Boundary: {self.lowerSlider.value()}")

    def updateUpperBound(self, value):
        if value <= self.lowerSlider.value():
            self.upperSlider.setValue(self.lowerSlider.value() + 1)
        self.upperBound = value
        self.upperSliderLabel.setText(f"Upper Boundary: {self.upperSlider.value()}")

    def updateProb(self, value):
        self.prob = value/100.0
        self.probSliderLabel.setText(f"Conn prob: {self.probSlider.value()/100.0:.2f}")

    def updateNetworkType(self, text):
        self.networktype = text

    def checkboxStateChanged(self):
        self.methods_list = []
        if self.checkbox1.isChecked():
            self.methods_list.append('Pearson Correlation')
        if self.checkbox2.isChecked():
            self.methods_list.append('Transfer Entropy')
        if self.checkbox3.isChecked():
            self.methods_list.append('CCG')
        if self.checkbox4.isChecked():
            self.methods_list.append('GLM')

    def checkboxWhiteNoiseChanged(self):
        self.whitenoise = False
        if self.checkbox_noise.isChecked():
            self.whitenoise = True

    def updateWeightCoef(self, value):
        self.weightsCoef = value
        self.weightsSliderLabel.setText(f"weights=coeff/N: {self.weightsSlider.value():.2f}")


    def updateDistrib(self, text):
        self.distrib = text



    def updatePlot(self):
        if (self.networktype is not None) and (self.upperBound is not None) and (
                self.lowerBound is not None) and (
                self.prob is not None) and (
                self.methods_list is not None) and (
                self.weightsCoef is not None) and (
                self.distrib is not None
        ):
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            if self.networktype == "ER":
                # ER_network.start_network_simulation(ax, self.prob,
                #                                     'iaf_psc_alpha',
                #                                     self.distrib,
                #                                     self.methods_list, self.weightsCoef, self.whitenoise,
                #                                     self.lowerBound, self.upperBound)

                pass
            elif self.networktype == "WS":
                pass
                #WS_network.start_network_simulation(ax, self.lowerBound, self.upperBound, self.prob, 'iaf_psc_alpha', 'Gaussian')

            self.canvas.draw()





    def launch(self):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        df = new_main.create_table(ax, draw=False)
        for method in ['Pearson Correlation', 'Transfer Entropy', 'CCG', 'GLM']:
            ax.plot(df[df['method']==method]['size'], df[df['method']==method]['f1'], label=method)
        ax.legend()
        self.canvas.draw()
        # self.updatePlot()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())


# 1 neuron
# motifs
# adjust parameters
# data driven paper - why this mechanism