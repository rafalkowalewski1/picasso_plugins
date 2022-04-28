"""
Picasso: Render 3D plugin to display points saved in a yaml file.

NOTE: may be incompatible with other plugins as it modifies the
	  draw_scene function in ViewRotation

Author: Rafal Kowalewski 2022
"""

import numpy as np
import yaml

from PyQt5 import QtCore, QtGui, QtWidgets

from ... import render

def is_hexadecimal(text):
    allowed_characters = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f',
        'A', 'B', 'C', 'D', 'E', 'F',
    ]
    sum_char = 0
    if type(text) == str:
        if text[0] == '#':
            if len(text) == 7:
                for char in text[1:]:
                    if char in allowed_characters:
                        sum_char += 1
                if sum_char == 6:
                    return True
    return False


class Plugin():
	def __init__(self, window):
		self.name = "render"
		self.window = window
		self.view_rot = window.window_rot.view_rot

	def execute(self):
		file_menu_rot = self.window.window_rot.menus[0]
		file_menu_rot.addSeparator()

		open_points_action = file_menu_rot.addAction("Load points")
		open_points_action.triggered.connect(self.open_points)

	def open_points(self):
		path, exe = QtWidgets.QFileDialog.getOpenFileName(
			self.window.window_rot, "Load points", filter="*.yaml"
		)
		if path:
			self.display_points(path)

	def display_points(self, path):
		with open(path, "r") as f:
			points = yaml.full_load(f)
		if not "Centers" in points:
			raise ValueError("Unrecognized points file")
		self._centers = np.asarray(
			points["Centers"]
		)
		try:
			self._centers_color = points["Color"]
			if self._centers_color == "gray":
				self._centers_color = "grey"
		except:
			pass
		self.view_rot.point_color_warning = True
		self.view_rot.draw_scene = self.draw_scene
		self.view_rot.update_scene()

	def draw_scene(self, viewport):
		self.view_rot.viewport = self.view_rot.adjust_viewport_to_view(viewport)
		qimage = self.view_rot.render_scene()
		self.view_rot.qimage = qimage.scaled(
			self.view_rot.width(),
			self.view_rot.height(),
			QtCore.Qt.KeepAspectRatioByExpanding,
		)
		self.view_rot.qimage = self.view_rot.draw_scalebar(
			self.view_rot.qimage
		)
		if self.view_rot.display_legend:
			self.view_rot.qimage = self.view_rot.draw_legend(
				self.view_rot.qimage
			)
		if self.view_rot.display_rotation:
			self.view_rot.qimage = self.view_rot.draw_rotation(
				self.view_rot.qimage
			)
		if len(self._centers) > 0:
			self.view_rot.qimage = self.draw_points_rotation(
				self.view_rot.qimage
			)
		self.view_rot.qimage = self.view_rot.draw_points(self.view_rot.qimage)
		self.view_rot.pixmap = QtGui.QPixmap.fromImage(self.view_rot.qimage)
		self.view_rot.setPixmap(self.view_rot.pixmap)

	def draw_points_rotation(self, image):
		painter = QtGui.QPainter(image)
		if type(self._centers_color) == str:
			if len(self._centers_color) > 0:
				if is_hexadecimal(self._centers_color):
					r = int(self._centers_color[1:3], 16) / 255.
					g = int(self._centers_color[3:5], 16) / 255.
					b = int(self._centers_color[5:], 16) / 255.
					painter.setBrush(
						QtGui.QBrush(QtGui.QColor.fromRgbF(r, g, b, 1))
					)
				elif self._centers_color in [
					"red", "yellow", "blue", "green", "white",
					"black", "cyan", "magenta", "gray",
				]:
					painter.setBrush(
						QtGui.QBrush(QtGui.QColor(self._centers_color))
					)
				else:
					self.print_warning()
					painter.setBrush(QtGui.QBrush(QtGui.QColor("red")))
			else:
				self.print_warning()
				painter.setBrush(QtGui.QBrush(QtGui.QColor("red")))
		else:
			self.print_warning()
			painter.setBrush(QtGui.QBrush(QtGui.QColor("red")))

		# translate the points to the origin (0, 0)
		(y_min, x_min), (y_max, x_max) = self.view_rot.viewport
		centers = self._centers.copy()
		centers[:, 0] -= x_min + (x_max - x_min) / 2
		centers[:, 1] -= y_min + (y_max - y_min) / 2
		centers[:, 2] /= self.window.display_settings_dlg.pixelsize.value()

		# rotate the points
		R = render.rotation_matrix(
			self.view_rot.angx, self.view_rot.angy, self.view_rot.angz
		)
		centers = R.apply(centers)

		# translate the points back to their original coords
		centers[:, 0] += x_min + (x_max - x_min) / 2
		centers[:, 1] += y_min + (y_max - y_min) / 2
		centers[:, 2] *= self.window.display_settings_dlg.pixelsize.value()

		# translate the points to the pixels of the display
		centers[:, 0] = (
			(centers[:, 0] - x_min) 
			* (self.view_rot.width() / (x_max - x_min))
		)
		centers[:, 1] = (
			(centers[:, 1] - y_min) 
			* (self.view_rot.height() / (y_max - y_min))
		)

		# draw the points
		for coord in centers:
			x = coord[0]
			y = coord[1]
			painter.drawEllipse(QtCore.QPoint(x, y), 3, 3)

		return image

	def print_warning(self):
		if self.view_rot.point_color_warning:
			message = (
				"Unrecognized color of the points.  The default color"
				" (red) will be used"
			)
			self.view_rot.point_color_warning = False
			QtWidgets.QMessageBox.information(
				self.view_rot, "Warning", message
			)