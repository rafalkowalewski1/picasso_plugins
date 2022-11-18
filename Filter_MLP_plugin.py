"""
Picasso: Render plugin implementing Nanotron: Predict in the render window.
Author: Rafal Kowalewski 2022
"""

import os

import numpy as np
import yaml

import joblib

from PyQt5 import QtWidgets

from ... import io, lib, nanotron


class Plugin():
    def __init__(self, window):
        self.name = "render"
        self.window = window

    def execute(self):
        tools_menu = self.window.menus[2]
        tools_menu.addSeparator()

        nanotron_filter_action = tools_menu.addAction(
            "Filter picks with an MLP"
        )
        nanotron_filter_action.triggered.connect(self.nanotron_filter)

    def nanotron_filter(self):
        #load the model
        if self.window.view._pick_shape != "Circle":
            raise ValueError(
                "The tool is compatible with circular picks only."
            )
        if self.window.view._picks == []:
            raise ValueError("No picks chosen, please pick first.")
        channel = self.window.view.get_channel("Choose channel to filter")
        if channel is not None:
            path, exe = QtWidgets.QFileDialog.getOpenFileName(
                self.window, "Load model file", filter="*.sav", directory=None
            )
            if path:
                try:
                    model = joblib.load(path)
                except Exception:
                    raise ValueError("No model file loaded.")
                try:
                    base, ext = os.path.splitext(path)
                    with open(base + ".yaml", "r") as f:
                        model_info = yaml.full_load(f)
                except io.NoMetadataFileError:
                    return
                self.window.filter_dialog = Filter_MLP_Dialog(
                    self.window, model, model_info, channel
                )
                self.window.dialogs.append(self.window.filter_dialog)
                self.window.filter_dialog.show()


class Filter_MLP_Dialog(QtWidgets.QDialog):
    def __init__(self, window, model, model_info, channel):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Filter picks with an MLP")
        self.setModal(False)
        self.model = model
        self.all_picks = self.window.view._picks
        self.classes = model_info["Classes"]
        self.pick_radius = model_info["Pick Diameter"] / 2
        self.oversampling = model_info["Oversampling"]
        self.channel = channel
        self.predictions = []
        self.probabilites = []
        self.to_keep = []

        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)

        self.classes_box = QtWidgets.QComboBox(self)
        for value in self.classes.values():
            self.classes_box.addItem(value)

        self.prob_thresh = QtWidgets.QDoubleSpinBox()
        self.prob_thresh.setDecimals(6)
        self.prob_thresh.setRange(0.0, 1.0)
        self.prob_thresh.setValue(0.995)
        self.prob_thresh.setSingleStep(0.000001)

        self.predict_button = QtWidgets.QPushButton("Predict")
        self.predict_button.clicked.connect(self.update_scene)

        self.layout.addWidget(QtWidgets.QLabel("Choose class:"), 0, 0)
        self.layout.addWidget(self.classes_box, 0, 1)
        self.layout.addWidget(QtWidgets.QLabel("Filter probability:"), 1, 0)
        self.layout.addWidget(self.prob_thresh, 1, 1)
        self.layout.addWidget(self.predict_button, 2, 1)

        self.predict()
        self.update_scene()

    def update_scene(self):
        self.update_picks()
        self.window.view.update_scene()
        self.window.info_dialog.n_picks.setText(str(len(self.to_keep)))
        self.to_keep = []

    def predict(self):
        l = lib.ProgressDialog(
            "Predicting structures...", 0, len(self.all_picks), self
        )
        l.show()
        l.set_value(0)
        to_delete = []
        for i in range(len(self.all_picks)):
            l.set_value(i)
            try:
                pred, prob = nanotron.predict_structure(
                    self.model,
                    self.window.view.locs[self.channel],
                    i,
                    self.pick_radius,
                    self.oversampling,
                    picks=self.all_picks[i],
                )
                self.predictions.append(pred[0])
                self.probabilites.append(prob[0])
            except:
                to_delete.append(i)
        l.close()
        if len(to_delete) != 0:
            for i in sorted(to_delete, reverse=True):
                del self.all_picks[i]

    def update_picks(self):
        # get the index of the currently chosen class
        # this is used later for indexing the predictions and probs
        classes_names = np.array(list(self.classes.values()))
        idx = np.where(classes_names == self.classes_box.currentText())[0][0]

        # find which picks are to be kept
        for i in range(len(self.all_picks)):
            check_prob = self.probabilites[i][idx] >= self.prob_thresh.value()
            check_class = self.classes[self.predictions[i]] == \
                self.classes_box.currentText()
            if check_prob and check_class:
                self.to_keep.append(i)
            self.window.view._picks = []
            for i in self.to_keep:
                self.window.view._picks.append(self.all_picks[i])
