# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:18:11 2019

@author: User
"""
import os
import sys

from PySide2 import QtGui, QtWidgets
from twitterScrape import twitterScrape

import main_gui

class MainWindow(QtWidgets.QMainWindow, main_gui.Ui_MainWindow):
	def __init__(self):
		super(self.__class__, self).__init__()
		
		self.setupUi(self)
		
		self.updateStatus("Welcome to Twitter sentiment analysis. Type in Twitter name.")
		
		self.statusBox.setReadOnly(True)
		self.bsLimit.setText('100000')
		
		self.outputDirButton.clicked.connect(self.onOutputDirectoryButton)
		self.analyzeButton.clicked.connect(self.onAnalyzeButton)
		self.actionQuit.triggered.connect(self.close)
		
		if getattr(sys, 'frozen', False):
			self.application_path = os.path.dirname(sys.argv[0])
		elif __file__:
			self.application_path = os.path.dirname(os.path.realpath(__file__))
		print(self.application_path)
		
	def onOutputDirectoryButton(self):
		dialog = QtWidgets.QFileDialog(self)
		dialog.setFileMode(QtWidgets.QFileDialog.Directory)
		if dialog.exec_():
			self.output_path = dialog.selectedFiles()[0]
			self.outputDirText.setText(self.output_path)
		
	def onAnalyzeButton(self):
		self.updateStatus("Scraping twitter...")
		
		# Set Qthread
		self.worker = twitterScrape()
		self.worker.company_name = self.twitterNameBox.text()
		self.worker.output_path = self.output_path
		self.worker.application_path = self.application_path
		self.worker.bsLimit = self.bsLimit.text()
		
		# Set Qthread signals
		self.worker.progressEvent.connect(self.statusBoxUpdate)
		self.worker.plotEvent.connect(self.plotEvent)
		self.worker.finished.connect(self.doneScraping)
		
		# Execute
		self.worker.start()
	
	def plotEvent(self, figure, idx):
		if idx ==1:
			graphicscene1 = QtWidgets.QGraphicsScene()
			graphicscene1.addWidget(figure)
			self.graphicsViewTab1.setScene(graphicscene1)
			self.graphicsViewTab1.show()
			figure.draw()
			QtGui.QGuiApplication.processEvents()
		elif idx ==2:
			graphicscene2 = QtWidgets.QGraphicsScene()
			graphicscene2.addWidget(figure)
			self.graphicsViewTab2.setScene(graphicscene2)
			self.graphicsViewTab2.show()
			figure.draw()
			QtGui.QGuiApplication.processEvents()
		elif idx ==3:
			graphicscene3 = QtWidgets.QGraphicsScene()
			graphicscene3.addWidget(figure)
			self.graphicsViewTab3.setScene(graphicscene3)
			self.graphicsViewTab3.show()
			figure.draw()
			QtGui.QGuiApplication.processEvents()
		
	def updateStatus(self, update):
		"""Updates the status bar located at the bottom of the window.
		param: update
			The message to be displayed.
		"""
		self.statusbar.showMessage(update)
	
	def statusBoxUpdate(self, text):
		self.statusBox.appendPlainText(text)
		QtGui.QGuiApplication.processEvents()
		
	def doneScraping(self):
		self.statusBox.appendPlainText('\nCompleted conversion of {}'.format(self.twitterNameBox.text()))
		self.updateStatus("Scraping complete.")
			
def main():
	app = QtWidgets.QApplication(sys.argv)
	window = MainWindow()
	window.show()
	app.exec_()
	
if __name__ == '__main__':	
	main()