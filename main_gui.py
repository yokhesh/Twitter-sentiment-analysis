# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_gui.ui',
# licensing of 'main_gui.ui' applies.
#
# Created: Sun Nov 24 22:14:17 2019
#      by: pyside2-uic  running on PySide2 5.13.1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(809, 943)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setContentsMargins(-1, 20, -1, 20)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_5 = QtWidgets.QWidget(self.centralwidget)
        self.widget_5.setObjectName("widget_5")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget_5)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.groupBox = QtWidgets.QGroupBox(self.widget_5)
        self.groupBox.setMaximumSize(QtCore.QSize(600, 16777215))
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_4 = QtWidgets.QWidget(self.groupBox)
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.outputDirButton = QtWidgets.QPushButton(self.widget_4)
        self.outputDirButton.setMinimumSize(QtCore.QSize(150, 0))
        self.outputDirButton.setMaximumSize(QtCore.QSize(150, 16777215))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 160))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 160))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 160))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        self.outputDirButton.setPalette(palette)
        self.outputDirButton.setDefault(True)
        self.outputDirButton.setObjectName("outputDirButton")
        self.horizontalLayout_2.addWidget(self.outputDirButton)
        self.outputDirText = QtWidgets.QLineEdit(self.widget_4)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setWeight(50)
        font.setBold(False)
        self.outputDirText.setFont(font)
        self.outputDirText.setObjectName("outputDirText")
        self.horizontalLayout_2.addWidget(self.outputDirText)
        self.verticalLayout.addWidget(self.widget_4)
        self.widget = QtWidgets.QWidget(self.groupBox)
        self.widget.setObjectName("widget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setMinimumSize(QtCore.QSize(150, 0))
        self.label.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 1, 1, 1)
        self.twitterNameBox = QtWidgets.QLineEdit(self.widget)
        self.twitterNameBox.setMinimumSize(QtCore.QSize(0, 0))
        self.twitterNameBox.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setWeight(50)
        font.setBold(False)
        self.twitterNameBox.setFont(font)
        self.twitterNameBox.setObjectName("twitterNameBox")
        self.gridLayout_4.addWidget(self.twitterNameBox, 0, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem, 0, 3, 1, 1)
        self.verticalLayout.addWidget(self.widget)
        self.widget_6 = QtWidgets.QWidget(self.groupBox)
        self.widget_6.setObjectName("widget_6")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_6)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.widget_6)
        self.label_3.setMinimumSize(QtCore.QSize(150, 0))
        self.label_3.setMaximumSize(QtCore.QSize(150, 16777215))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.bsLimit = QtWidgets.QLineEdit(self.widget_6)
        self.bsLimit.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setWeight(50)
        font.setBold(False)
        self.bsLimit.setFont(font)
        self.bsLimit.setObjectName("bsLimit")
        self.horizontalLayout_3.addWidget(self.bsLimit)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.verticalLayout.addWidget(self.widget_6)
        self.gridLayout_3.addWidget(self.groupBox, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.widget_5)
        self.widget_3 = QtWidgets.QWidget(self.centralwidget)
        self.widget_3.setObjectName("widget_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget_3)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.analyzeButton = QtWidgets.QPushButton(self.widget_3)
        self.analyzeButton.setMaximumSize(QtCore.QSize(200, 16777215))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 160))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 160))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 160))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        self.analyzeButton.setPalette(palette)
        self.analyzeButton.setAutoDefault(False)
        self.analyzeButton.setDefault(True)
        self.analyzeButton.setFlat(False)
        self.analyzeButton.setObjectName("analyzeButton")
        self.gridLayout_2.addWidget(self.analyzeButton, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.widget_3)
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setMinimumSize(QtCore.QSize(0, 150))
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 150))
        self.widget_2.setObjectName("widget_2")
        self.gridLayout = QtWidgets.QGridLayout(self.widget_2)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.statusBox = QtWidgets.QPlainTextEdit(self.widget_2)
        self.statusBox.setMaximumSize(QtCore.QSize(600, 150))
        self.statusBox.setObjectName("statusBox")
        self.gridLayout.addWidget(self.statusBox, 1, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.tab1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.graphicsViewTab1 = QtWidgets.QGraphicsView(self.tab1)
        self.graphicsViewTab1.setObjectName("graphicsViewTab1")
        self.horizontalLayout.addWidget(self.graphicsViewTab1)
        self.tabWidget.addTab(self.tab1, "")
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setObjectName("tab2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.tab2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.graphicsViewTab2 = QtWidgets.QGraphicsView(self.tab2)
        self.graphicsViewTab2.setObjectName("graphicsViewTab2")
        self.horizontalLayout_4.addWidget(self.graphicsViewTab2)
        self.tabWidget.addTab(self.tab2, "")
        self.tab3 = QtWidgets.QWidget()
        self.tab3.setObjectName("tab3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.tab3)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.graphicsViewTab3 = QtWidgets.QGraphicsView(self.tab3)
        self.graphicsViewTab3.setObjectName("graphicsViewTab3")
        self.horizontalLayout_5.addWidget(self.graphicsViewTab3)
        self.tabWidget.addTab(self.tab3, "")
        self.verticalLayout_2.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 809, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "Twitter Sentiment Analysis", None, -1))
        self.groupBox.setTitle(QtWidgets.QApplication.translate("MainWindow", "Settings", None, -1))
        self.outputDirButton.setText(QtWidgets.QApplication.translate("MainWindow", "Output Directory", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("MainWindow", "Twitter Account", None, -1))
        self.label_3.setText(QtWidgets.QApplication.translate("MainWindow", "Limit", None, -1))
        self.analyzeButton.setText(QtWidgets.QApplication.translate("MainWindow", "Analyze", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("MainWindow", "Status", None, -1))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1), QtWidgets.QApplication.translate("MainWindow", "Tweet Distribution", None, -1))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab2), QtWidgets.QApplication.translate("MainWindow", "Positive", None, -1))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab3), QtWidgets.QApplication.translate("MainWindow", "Negative", None, -1))
        self.menuFile.setTitle(QtWidgets.QApplication.translate("MainWindow", "File", None, -1))
        self.actionQuit.setText(QtWidgets.QApplication.translate("MainWindow", "Quit", None, -1))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

