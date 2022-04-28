"""
Picasso: Render plugin to calculate Ripley's H function in a single pick.

Based on: Khater IM, Nabi IR, Hamarneh G. 
	"A review of super-resolution single-molecule localization microscopy 
	 cluster analysis and quantification methods."
	Patterns. 2020 Jun 12;1(3):100038.

Author: Rafal Kowalewski 2022
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PyQt5 import QtGui, QtWidgets 

class Plugin():
	def __init__(self, window):
		self.name = "render"
		self.window = window

	def execute(self):
		postprocess_menu = self.window.menus[3]
		postprocess_menu.addSeparator()

		ripley_action = postprocess_menu.addAction("Ripley H function")
		ripley_action.triggered.connect(self.ripley)

	def ripley(self):
		channel = self.window.view.get_channel("Ripley H function")
		if channel is not None:
			# extract locs from the pick
			if len(self.window.view._picks) != 1:
				message = (
					"Ripley H function works well only on small areas, "
					" please pick only one region."
				)
				QtWidgets.QMessageBox.information(
					self.window.view, "Warning", message
				)
			else:
				locs = self.window.view.picked_locs(channel)

			# show the plot
			self.window.view.ripley_window = RipleyPlotWindow(self.window.view)
			self.window.view.ripley_window.plot(locs[0])
			self.window.view.ripley_window.show()


class RipleyPlotWindow(QtWidgets.QTabWidget):
	def __init__(self, view):
		super().__init__()
		self.setWindowTitle("Ripley H Function Plot")
		this_dir = os.path.dirname(os.path.realpath(__file__))
		icon_path = os.path.join(this_dir, "icons", "render.ico")
		icon = QtGui.QIcon(icon_path)
		self.setWindowIcon(icon)
		self.resize(500, 500)
		self.view = view
		self.figure = plt.Figure()
		self.canvas = FigureCanvas(self.figure)
		vbox = QtWidgets.QVBoxLayout()
		self.setLayout(vbox)
		vbox.addWidget(self.canvas)
		vbox.addWidget((NavigationToolbar2QT(self.canvas, self)))

	def plot(self, locs):
		self.figure.clear()
		p = self.view.window.display_settings_dlg.pixelsize.value()

		# calculate ripley's h function
		x = locs.x
		y = locs.y
		A = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y)) * (p ** 2)
		n = len(locs)

		radius = np.linspace(0, 50, 1000) # [nm]
		L = np.zeros(len(radius))
		points = np.stack((x, y)).T
		distance = distance_matrix(points, points) * p
		for i, r in enumerate(radius):
			L[i] = len(np.where(distance < r)[0])
		L = np.sqrt(A * L / (np.pi * n * (n - 1)))
		H = L - radius

		# plot
		ax = self.figure.add_subplot(111)
		ax.plot(radius, H, c='blue')
		ax.set_xlabel("distance [nm]")
		ax.set_ylabel("H function")

		self.canvas.draw()