"""
Picasso plugin to perform binding kinetics barcodes classification.

Assumes that one .hdf5 file is loaded with 20 nm grids.

The user needs to picks grids of interest and save them using
"Save pick regions".

Also, barcodes localizations are needed.

This is a Picasso pluing and is not compatible with the current 
official version. To enable it, clone Picasso from my fork using the
same instructions as in the official website: 
https://github.com/jungmannlab/picasso, except clone the directory with
git clone https://github.com/rafalkowalewski1/picasso.git -b development

Then add this python script into picasso/picasso/gui/plugins.

Author: Rafal Kowalewski
Date: April 2022
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.spatial import distance_matrix
from numba import njit
from numpy.lib.recfunctions import stack_arrays

# from icecream import ic

from PyQt5 import QtCore, QtGui, QtWidgets

from ... import io, lib, render, postprocess

# what fraction of locs must be clustered for each barcode type to be
# classified
FRACTIONS = [.65, .55, .3, .55, .4, .6]
# ranges of allowed distances between each pair of positions in barcodes;
# distances are sorted by ascending order, that is:
# 1-2; 2-3, 3-4 (same); 1-3; 2-4; 1-4
# /all in nm
DISTANCES = [
    [24.05, 34.58, 34.58, 47.97, 59.02, 63.96], # min
    [40.95, 47.97, 47.97, 66.04, 79.04, 82.94], # max
]
# indeces for each barcode, indicating which position pairs they possess
# for example, R5, has positions 2, 3 and 4, thus it has pairs 2-3, 3-4
# and 2-4, which correspond to indeces [1, 2, 4] in DISTANCES
R2_IDX = [0]
R3_IDX = [0, 1, 2, 3, 4, 5]
R4_IDX = [5]
R5_IDX = [1, 2, 4]
R6_IDX = [0, 4, 5]


# clustering functions
@njit 
def count_neighbors(dist, eps):
    nn = np.zeros(dist.shape[0], dtype=np.int32)
    for i in range(len(nn)):
        nn[i] = np.where(dist[i] <= eps)[0].shape[0] - 1
    return nn

@njit
def local_maxima(dist, nn, eps):
    """
    Finds the positions of the local maxima which are the locs 
    with the highest number of neighbors within given distance dist
    """
    n = dist.shape[0]
    lm = np.zeros(n, dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if dist[i][j] <= eps and nn[i] >= nn[j]:
                lm[i] = 1
            if dist[i][j] <= eps and nn[i] < nn[j]:
                lm[i] = 0
                break
    return lm

@njit
def assign_to_cluster(i, dist, eps, cluster_id):
    n = dist.shape[0]
    for j in range(n):
        if dist[i][j] <= eps:
            if cluster_id[i] != 0:
                if cluster_id[j] == 0:
                    cluster_id[j] = cluster_id[i]
            if cluster_id[i] == 0:
                if j == 0:
                    cluster_id[i] = i + 1
                cluster_id[j] = i + 1
    return cluster_id

@njit
def check_cluster_size(cluster_n_locs, min_samples, cluster_id):
    for i in range(len(cluster_id)):
        if cluster_n_locs[cluster_id[i]] <= min_samples:
            cluster_id[i] = 0
    return cluster_id

@njit
def rename_clusters(cluster_id, clusters):
    for i in range(len(cluster_id)):
        for j in range(len(clusters)):
            if cluster_id[i] == clusters[j]:
                cluster_id[i] = j
    return cluster_id

@njit 
def cluster_properties(cluster_id, n_clusters, frame):
    mean_frame = np.zeros(n_clusters, dtype=np.float32)
    n_locs_cluster = np.zeros(n_clusters, dtype=np.int32)
    locs_in_window = np.zeros((n_clusters, 21), dtype=np.int32)
    locs_perc = np.zeros(n_clusters, dtype=np.float32)
    window_search = frame[-1] / 20
    for j in range(n_clusters):
        for i in range(len(cluster_id)):
            if j == cluster_id[i]:
                n_locs_cluster[j] += 1
                mean_frame[j] += frame[i]
                locs_in_window[j][int(frame[i] / window_search)] += 1
    mean_frame = mean_frame / n_locs_cluster
    for i in range(n_clusters):
        for j in range(21):
            temp = locs_in_window[i][j] / n_locs_cluster[i]
            if temp > locs_perc[i]:
                locs_perc[i] = temp
    return mean_frame, locs_perc

@njit
def find_true_clusters(mean_frame, locs_perc, frame):
    n_frame = np.int32(np.max(frame))
    true_cluster = np.zeros(len(mean_frame), dtype=np.int8)
    for i in range(len(mean_frame)):
        cond1 = locs_perc[i] < 0.8
        cond2 = mean_frame[i] < n_frame * 0.8
        cond3 = mean_frame[i] > n_frame * 0.2
        if cond1 and cond2 and cond3:
            true_cluster[i] = 1
    return true_cluster

def cluster(x, y, frame, eps, min_samples):
    xy = np.stack((x, y)).T
    cluster_id = np.zeros(len(x), dtype=np.int32)
    d = distance_matrix(xy, xy)
    n_neighbors = count_neighbors(d, eps)
    local_max = local_maxima(d, n_neighbors, eps)
    for i in range(len(x)):
        if local_max[i]:
            cluster_id = assign_to_cluster(i, d, eps, cluster_id)
    cluster_n_locs = np.bincount(cluster_id)
    cluster_id = check_cluster_size(
        cluster_n_locs, min_samples, cluster_id
    )
    clusters = np.unique(cluster_id)
    n_clusters = len(clusters)
    cluster_id = rename_clusters(cluster_id, clusters)
    mean_frame, locs_perc = cluster_properties(
        cluster_id, n_clusters, frame
    )
    true_cluster = find_true_clusters(mean_frame, locs_perc, frame)
    labels = -1 * np.ones(len(x), dtype=np.int8)
    for i in range(len(x)):
        if (cluster_id[i] != 0) and (true_cluster[cluster_id[i]] == 1):
            labels[i] = cluster_id[i] - 1
    return labels
# end of clustering functions

class MainDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setWindowTitle("Barcodes classification")
        self.parameters_changed = False
        self.first_test = True
        self.ready_to_display = False
        self.index_blocks = None
        self.cluster_success = []
        self.classification_results = []
        self.barcodes_settings = BarcodesSettings(self)
        self.view = ClusteringView(self)
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)

        # Load data
        load_data_box = QtWidgets.QGroupBox("Load data")
        self.layout.addWidget(load_data_box, 0, 0)
        load_data_grid = QtWidgets.QGridLayout(load_data_box)
        # Load data - pick regions with grids
        self.get_picks = QtWidgets.QPushButton(
            "Choose pick regions\nwith grids"
        )
        self.get_picks.clicked.connect(self.get_picks_path)
        load_data_grid.addWidget(self.get_picks, 0, 0)
        self.picks_path = QtWidgets.QLabel("None")
        load_data_grid.addWidget(self.picks_path, 0, 1)
        # Load data - .hdf5 file with barcodes
        self.get_barcodes = QtWidgets.QPushButton(
            "Choose .hdf5\nwith barcodes"
        )
        load_data_grid.addWidget(self.get_barcodes, 1, 0)
        self.get_barcodes.clicked.connect(self.get_barcodes_path)
        self.barcodes_path = QtWidgets.QLabel("None")
        load_data_grid.addWidget(self.barcodes_path, 1, 1)

        # Barcodes classification test
        classification_box = QtWidgets.QGroupBox(
            "Test barcodes classification"
        )
        self.layout.addWidget(classification_box, 1, 0)
        classification_grid = QtWidgets.QGridLayout(classification_box)
        # Barcodes classification test - cluster radius
        classification_grid.addWidget(
            QtWidgets.QLabel("Cluster radius [nm]: "), 0, 0
        )
        self.radius = QtWidgets.QDoubleSpinBox(self)
        self.radius.setKeyboardTracking(True)
        self.radius.setDecimals(1)
        self.radius.setMinimum(5.0)
        self.radius.setMaximum(15.0)
        self.radius.setValue(10.0)
        self.radius.valueChanged.connect(self.on_parameters_changed)
        classification_grid.addWidget(self.radius, 0, 1)
        # Barcodes classification test - min cluster size
        classification_grid.addWidget(
            QtWidgets.QLabel("Min cluster size: "), 1, 0
        )
        self.cluster_size = QtWidgets.QSpinBox(self)
        self.cluster_size.setKeyboardTracking(True)
        self.cluster_size.setMinimum(1)
        self.cluster_size.setMaximum(5e3)
        self.cluster_size.setValue(40)
        self.cluster_size.valueChanged.connect(self.on_parameters_changed)
        classification_grid.addWidget(self.cluster_size, 1, 1)
        # Barcodes classification test - number of picks for testing
        classification_grid.addWidget(
            QtWidgets.QLabel("Number of barcodes\nfor testing"), 2, 0
        )
        self.n_test = QtWidgets.QSpinBox(self)
        self.n_test.setKeyboardTracking(True)
        self.n_test.setMinimum(3)
        self.n_test.setMaximum(10)
        self.n_test.setValue(5)
        self.n_test.valueChanged.connect(self.on_n_barcodes_changed)
        classification_grid.addWidget(self.n_test, 2, 1)
        # Barcodes classification test - display results info
        self.show_class_name = QtWidgets.QCheckBox("Display names")
        self.show_class_name.setChecked(True)
        self.show_class_name.stateChanged.connect(self.view.update_scene)
        classification_grid.addWidget(self.show_class_name, 3, 0)
        self.show_cluster_success = QtWidgets.QCheckBox(
            "Display cluster success"
        )
        self.show_cluster_success.stateChanged.connect(self.view.update_scene)
        classification_grid.addWidget(self.show_cluster_success, 4, 0)
        # Barcodes classification test - more settings
        self.more_settigns = QtWidgets.QPushButton("More settings")
        self.more_settigns.clicked.connect(self.barcodes_settings.show)
        classification_grid.addWidget(self.more_settigns, 5, 0)     
        # Barcodes classification test - test
        self.test_barcodes = QtWidgets.QPushButton("Test")
        self.test_barcodes.setDefault(True)
        self.test_barcodes.clicked.connect(self.test)
        classification_grid.addWidget(self.test_barcodes, 5, 1)

        # Barcodes classification all
        c_box = QtWidgets.QGroupBox("Barcodes classification")
        self.layout.addWidget(c_box, 2, 0)
        c_grid = QtWidgets.QGridLayout(c_box)
        # Barcodes classification all - save pick regions
        self.save_pick_regions = QtWidgets.QCheckBox(
            "Save classified picks"
        )
        c_grid.addWidget(self.save_pick_regions, 0, 0)
        # Barcodes classification all - save picked grids
        self.save_picked_grids = QtWidgets.QCheckBox(
            "Save classified grids"
        )
        c_grid.addWidget(self.save_picked_grids, 1, 0)
        # Barcodes classification all - classify all barcodes
        self.classify_barcodes = QtWidgets.QPushButton("Classify all barcodes")
        self.classify_barcodes.setDefault(False)
        self.classify_barcodes.clicked.connect(self.classify_all)
        c_grid.addWidget(self.classify_barcodes, 2, 0)
        # Barcodes classification all - show results
        for i in range(7):
            if i != 6:
                line = f"R{i+1}: "
            else:
                line = "Unclassified: "
            widget = QtWidgets.QLabel(line)
            self.classification_results.append(widget)
            c_grid.addWidget(widget, i+3, 0)

        # Barcodes clustering insight
        view_box = QtWidgets.QGroupBox("Test classification")
        self.layout.addWidget(view_box, 0, 1, 2, 1)
        view_grid = QtWidgets.QGridLayout(view_box)
        view_grid.addWidget(self.view)

        # Cluster and classify all binding sites / todo 

    def on_parameters_changed(self):
        self.parameters_changed = True

    def on_n_barcodes_changed(self):
        self.first_test = True

    def get_picks_path(self):
        path, ext = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "Load pick regions", 
            directory=self.window.pwd, 
            filter="*.yaml",
        )
        if path:
            with open(path, "r") as f:
                regions = yaml.full_load(f)
                if "Centers" in regions:
                    self.picks = regions["Centers"]
                    self.pick_diameter = regions["Diameter"]
                    self.dx = 1.4 * self.pick_diameter
                    self.dy = 1.4 * self.pick_diameter
                    base, filename = os.path.split(path)
                    self.picks_path.setText(filename)
                    self.picks_full_path = path
                    self.index_blocks = None
                    self.ready_to_display = False
                else:
                    message = (
                        "Only circular picks allowed."
                    )
                    QtWidgets.QMessageBox.information(self, "Warning", message)

    def get_barcodes_path(self):
        path, ext = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "Load barcodes", 
            directory=self.window.pwd, 
            filter="*.hdf5",
        )
        if path:
            try:
                locs, info = io.load_locs(path)
            except io.NoMetadataFileError:
                return
            locs = lib.ensure_sanity(locs, info)
            self.all_barcodes = locs
            self.barcodes_info = info
            base, filename = os.path.split(path)
            self.barcodes_path.setText(filename)
            self.barcodes_full_path = path
            self.index_blocks = None
            self.ready_to_display = False

    def update_picks(self):
        # get new pick regions at random 
        test_idx = np.random.choice(
            np.arange(len(self.picks), dtype=np.int32),
            size=self.n_test.value() ** 2,
            replace=False,
        ) # random indeces
        self.test_picks = []
        for idx in test_idx:
            self.test_picks.append(self.picks[idx])

    def test(self):
        if not self.parameters_changed or self.first_test:
            self.update_picks()
        self.parameters_changed = False 

        picked_barcodes = self.picked_locs(self.test_picks)
        picked_clustered, self.cluster_success, self.test_names = (
            self.cluster_picks(picked_barcodes) 
        )
        self.barcodes = self.shift_locs(
            self.test_picks, picked_barcodes, save_centers=True
        )
        self.clustered_locs = self.shift_locs(
            self.test_picks, picked_clustered
        )
        if not self.ready_to_display:
            self.ready_to_display = True
        self.view.update_scene()
        self.first_test = False

    def cluster_picks(self, picked_locs):
        cluster_success = []
        names = []
        picked_clustered = []
        pixelsize = self.window.display_settings_dlg.pixelsize.value()
        radius_px = self.radius.value() / pixelsize
        progress = lib.ProgressDialog(
            "Clustering barcodes...", 0, len(picked_locs), self
        )
        progress.setValue(0)
        for i in range(len(picked_locs)):
            current_locs = picked_locs[i]
            temp_locs = []
            c_success = 0.0
            name = "None"
            if len(current_locs) > 0:
                labels = cluster(
                    current_locs.x, 
                    current_locs.y, 
                    current_locs.frame,
                    radius_px, 
                    self.cluster_size.value(),
                )
                if len(labels) > 0:
                    temp_locs = lib.append_to_rec(
                        current_locs, labels, "cluster"
                    )
                    n_old = len(temp_locs)
                    # discard locs that were unclustered
                    temp_locs = temp_locs[temp_locs.cluster != -1]
                    n_new = len(temp_locs)
                    c_success = n_new / n_old
                    idx = self.classify(
                        temp_locs,
                        c_success,
                        pixelsize
                    )
                    if idx is None:
                        name = "None"
                    else:
                        name = str(
                            self.barcodes_settings.barcodes_names[idx].text()
                        )
            cluster_success.append(c_success)
            names.append(name)
            picked_clustered.append(temp_locs)
            progress.setValue(i+1)
        progress.close()
        return picked_clustered, cluster_success, names

    def check_R1(self, c_success):
        R1_threshold = self.barcodes_settings.fraction_values[0].value()
        if self.barcodes_settings.barcodes_checks[0].isChecked():
            if c_success >= R1_threshold:
                return 0
        return None

    def check_R2_R4(self, d, c_success):
        R2_threshold = self.barcodes_settings.fraction_values[1].value()
        R4_threshold = self.barcodes_settings.fraction_values[3].value()
        if self.barcodes_settings.barcodes_checks[1].isChecked():
            if all([
                (d[i] >= DISTANCES[0][j]) and (d[i] <= DISTANCES[1][j])
                for i, j in enumerate(R2_IDX)
            ]):
                if c_success >= R2_threshold:
                    return 1
        if self.barcodes_settings.barcodes_checks[3].isChecked():
            if all([
                (d[i] >= DISTANCES[0][j]) and (d[i] <= DISTANCES[1][j])
                for i, j in enumerate(R4_IDX)
            ]):
                if c_success >= R4_threshold:
                    return 3
        return None

    def check_R3(self, d, c_success):
        R3_threshold = self.barcodes_settings.fraction_values[2].value()
        if self.barcodes_settings.barcodes_checks[2].isChecked():
            if all([
                (d[i] >= DISTANCES[0][j]) and (d[i] <= DISTANCES[1][j])
                for i, j in enumerate(R3_IDX)
            ]):
                if c_success >= R3_threshold:
                    return 2
        return None

    def check_R5_R6(self, d, c_success):
        R5_threshold = self.barcodes_settings.fraction_values[4].value()
        R6_threshold = self.barcodes_settings.fraction_values[5].value()
        if self.barcodes_settings.barcodes_checks[4].isChecked():
            if all([
                (d[i] >= DISTANCES[0][j]) and (d[i] <= DISTANCES[1][j])
                for i, j in enumerate(R5_IDX)
            ]):
                if c_success >= R5_threshold:
                    return 4
        if self.barcodes_settings.barcodes_checks[5].isChecked():
            if all([
                (d[i] >= DISTANCES[0][j]) and (d[i] <= DISTANCES[1][j])
                for i, j in enumerate(R6_IDX)
            ]):
                if c_success >= R6_threshold:
                    return 5
        return None

    def classify(self, locs, cluster_success, pixelsize):
        """ 
        classifies the given locs in picks after they're clustered

        finds distances between clusters, compares them to DISTANCES
        and verifies cluster success

            * locs - clustered locs
            * cluster_success - fraction of locs that have been clustered
            * pixelsize - camera pixel's size in nm
        """

        result = None
        n_clusters = len(np.unique(locs.cluster))

        # if only one cluster, check that enough locs have been clustered
        if n_clusters == 1:
            result = self.check_R1(cluster_success)
            return result

        # measure distances between clusters if there are 2, 3 or 4 of them
        elif n_clusters in [2, 3, 4]:
            # find centers of mass
            centers = np.zeros((n_clusters, 2), dtype=np.float32)
            cluster_ids = np.unique(locs.cluster)
            cluster_ids = np.delete(
                cluster_ids, np.where(cluster_ids == -1)[0]
            ) # todo: is this line needed?
            for j, cluster_id in enumerate(cluster_ids):
                c_locs = locs[locs.cluster == cluster_id]
                com_x = np.mean(c_locs.x)
                com_y = np.mean(c_locs.y)
                centers[j, :] = com_x, com_y

            # measure distances between clusters
            d = distance_matrix(centers, centers)
            d = np.unique(d)[1:] # dump the 0.0 distance
            d *= pixelsize

            # classify
            if n_clusters == 2:
                result = self.check_R2_R4(d, cluster_success)
                return result

            elif n_clusters == 3:
                result = self.check_R5_R6(d, cluster_success)
                return result

            else: # 4 clusters
                result = self.check_R3(d, cluster_success)
                return result

        return result

    def classify_all(self):
        """
        clusters all picks (barcodes) and classifies them, saves pick 
        regions
        """

        # pick all barcodes
        status = lib.StatusDialog("Picking all barcodes...", self)
        picked_barcodes = self.picked_locs(self.picks)
        status.close()

        # cluster them all
        _, _, names = self.cluster_picks(picked_barcodes)

        # todo: save pick regions as a variable I guess? it will be needed later

        # save pick regions and picked grids 
        if self.save_pick_regions.isChecked():
            self.save_all_picks(names)
        if self.save_picked_grids.isChecked():
            self.save_all_grids(names)

        # update classification results
        self.update_classification_info(names)

    def save_all_picks(self, names):
        base, _ = os.path.splitext(self.barcodes_full_path)
        for i in range(len(self.barcodes_settings.barcodes_checks)):
            if self.barcodes_settings.barcodes_checks[i].isChecked():
                picks = []
                for j, name in enumerate(names):
                    if name == str(
                            self.barcodes_settings.barcodes_names[i].text()
                        ):
                        picks.append(self.picks[j])
                all_data = {
                    "Diameter": float(self.pick_diameter),
                    "Centers": [
                        [float(_[0]), float(_[1])] for _ in picks
                    ],
                    "Shape": "Circle",
                }
                out_path = (
                    base 
                    + "_"
                    + self.barcodes_settings.barcodes_names[i].text()
                    + ".yaml"
                )
                with open(out_path, "w") as f:
                    yaml.dump(all_data, f)

    def save_all_grids(self, names):
        base, _ = os.path.splitext(self.window.view.locs_paths[0])
        r = self.pick_diameter / 2
        status = lib.StatusDialog("Indexing grids localizations...", self)
        if self.window.view.index_blocks[0] is None:
            index_blocks = postprocess.get_index_blocks(
                self.window.view.locs[0],
                self.window.view.infos[0],
                r,
            )
        else:
            index_blocks = self.window.view.index_blocks[0]
        status.close()
        for i in range(len(self.barcodes_settings.barcodes_checks)):
            if self.barcodes_settings.barcodes_checks[i].isChecked():
                picks = []
                for j, name in enumerate(names):
                    if name == str(
                            self.barcodes_settings.barcodes_names[i].text()
                        ):
                        picks.append(self.picks[j])
                # pick grids
                grids = []
                for pick in picks:
                    x, y = pick
                    block_locs = postprocess.get_block_locs_at(
                        x, y, index_blocks
                    )
                    locs_picked = lib.locs_at(x, y, block_locs, r)
                    locs_picked.sort(kind="mergesort", order="frame")
                    grids.append(locs_picked)
                # save them
                grids = stack_arrays(grids, asrecarray=True, usemask=False)
                new_info = {
                    "Generated by": "Picasso Render Barcodes",
                    "Pick Shape": "Circle",
                    "Pick Diameter": self.pick_diameter,
                    "Barcode ID": f"R{i+1}",
                    "Barcode Name": (
                        self.barcodes_settings.barcodes_names[i].text()
                    ),
                    "Number of picks": len(picks),
                }
                out_path = (
                    base 
                    + "_"
                    + self.barcodes_settings.barcodes_names[i].text()
                    + ".hdf5"
                )
                io.save_locs(
                    out_path, grids, self.window.view.infos[0] + [new_info]
                )

    def update_classification_info(self, names):
        barcodes_names = [
            _.text() for _ in self.barcodes_settings.barcodes_names
        ]
        names_arr = np.array(names, dtype=object)
        lengths = [
            len(np.where(names_arr == _)[0]) for _ in barcodes_names 
        ]
        lengths.append(len(self.picks) - np.sum(lengths[:-1]))
        for i in range(len(self.barcodes_settings.barcodes_names)):
            if self.barcodes_settings.barcodes_checks[i].isChecked():
                line = barcodes_names[i] + f": {lengths[i]}"
                self.classification_results[i].setText(line)
            else:
                line = barcodes_names[i] + ": Not Included"
                self.classification_results[i].setText(line)
        self.classification_results[-1].setText(f"Unclassified: {lengths[-1]}")

    def picked_locs(self, picks):
        if self.index_blocks is None:
            self.index_locs()

        picked_locs = []
        r = self.pick_diameter / 2

        for pick in picks:
            x, y = pick
            block_locs = postprocess.get_block_locs_at(x, y, self.index_blocks)
            locs_picked = lib.locs_at(x, y, block_locs, r)
            locs_picked.sort(kind="mergesort", order="frame")
            picked_locs.append(locs_picked)

        return picked_locs

    def index_locs(self):
        status = lib.StatusDialog("Indexing localizations...", self)
        self.index_blocks = postprocess.get_index_blocks(
            self.all_barcodes,
            self.barcodes_info,
            self.pick_diameter / 2,
        )
        status.close()

    def shift_locs(self, picks, locs, save_centers=False):
        """ shifts picked locs onto a rectangular grid. """
        if save_centers:
            self.centers = []

        # shift all picked locs to the origin (0, 0)
        for i, loc in enumerate(locs):
            x, y = picks[i]
            loc.x -= x
            loc.y -= y
            locs[i] = loc

        # shift picks onto a grid (like group unfolding in render)
        grid_length = int(self.n_test.value())
        for i, loc in enumerate(locs):
            shift_x = (np.mod(i, grid_length) + 1) * self.dx
            loc.x += shift_x
            shift_y = (np.floor(i / grid_length) + 1) * self.dy
            loc.y += shift_y
            locs[i] = loc
            if save_centers:
                self.centers.append([shift_x, shift_y])

        # stack locs from the list
        locs = stack_arrays(
            locs, asrecarray=True, usemask=False
        )

        return locs


class BarcodesSettings(QtWidgets.QDialog):
    def __init__(self, main_dialog):
        super().__init__()
        self.main_dialog = main_dialog
        self.setWindowTitle("Barcodes classification settings")
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)
        self.barcodes_checks = []
        self.barcodes_names = []
        self.fraction_values = []

        # Barcodes names
        names_box = QtWidgets.QGroupBox("Barcodes names")
        self.layout.addWidget(names_box)
        general_grid = QtWidgets.QGridLayout(names_box)
        for i in range(6):
            # check boxes
            current_check = QtWidgets.QCheckBox(f"R{i+1}: ")
            current_check.setChecked(True)
            current_check.stateChanged.connect(
                self.main_dialog.on_parameters_changed
            )
            self.barcodes_checks.append(current_check)
            general_grid.addWidget(current_check, i, 0)
            # barcodes' names
            current_name = QtWidgets.QLineEdit(f"R{i+1}")
            current_name.textChanged.connect(
                self.main_dialog.on_parameters_changed
            )
            self.barcodes_names.append(current_name)
            general_grid.addWidget(current_name, i, 1)

        # Cluster success for each barcode
        fractions_box = QtWidgets.QGroupBox("Cluster success")
        self.layout.addWidget(fractions_box)
        fractions_grid = QtWidgets.QGridLayout(fractions_box)
        for i in range(6):
            # which barcode
            fractions_grid.addWidget(QtWidgets.QLabel(f"R{i+1}:" ), i, 0)
            # threshold fraction
            current_fraction = QtWidgets.QDoubleSpinBox()
            current_fraction.setMinimum(0.01)
            current_fraction.setMaximum(0.99)
            current_fraction.setSingleStep(0.01)
            current_fraction.setDecimals(2)
            current_fraction.setValue(FRACTIONS[i])
            current_fraction.setKeyboardTracking(False)
            current_fraction.valueChanged.connect(
                self.main_dialog.on_parameters_changed
            )
            self.fraction_values.append(current_fraction)
            fractions_grid.addWidget(current_fraction, i, 1)
        help_button = QtWidgets.QPushButton("Help")
        help_button.clicked.connect(self.help_message)
        fractions_grid.addWidget(help_button)

    def help_message(self):
        message = (
            "For each barcode type,  choose the minimum fraction of"
            " localizations that must be clustered in each pick to be"
            " classified as a given barcode type."
            "\n\n"
            "For example,  if you take the value 0.70 for R2,  at least 70%"
            " of localizations in a given pick must be assigned to clusters"
            " to classify the pick as R2 (assuming the distance between the"
            " clusters is right)."
        )
        QtWidgets.QMessageBox.information(self, "Help", message)


class ClusteringView(QtWidgets.QLabel):
    def __init__(self, main_dialog):
        super().__init__()
        self.main_dialog = main_dialog
        self.setMinimumSize(600, 600)
        self.setMaximumSize(600, 600)

    def map_to_view(self, position):
        # position is a tuple of x and y
        x, y = position
        cx = self.width() * x / self.viewport[1][1]
        cy = self.height() * y / self.viewport[1][0]
        return cx, cy

    def update_scene(self):
        if self.main_dialog.ready_to_display:

            # get viewport
            x_min = np.min(self.main_dialog.barcodes.x) - 0.2
            x_max = (
                np.max(self.main_dialog.barcodes.x) 
                + self.main_dialog.pick_diameter
            )
            y_min = np.min(self.main_dialog.barcodes.y) + 0.2
            y_max = (
                np.max(self.main_dialog.barcodes.y) 
                + self.main_dialog.pick_diameter
            )
            self.viewport = [(0, 0), (y_max, x_max)] 

            # get optimal oversampling
            os_horizontal = self.width() / (x_max - x_min)
            os_vertical = self.height() / (y_max - y_min)
            optimal_oversampling = max(os_horizontal, os_vertical)

            # render the barcodes locs
            _, image_locs = render.render(
                self.main_dialog.barcodes,
                oversampling=optimal_oversampling,
                viewport=self.viewport,
                blur_method="convolve",
            ) 
            # render the clustered locs
            _, image_clustered = render.render(
                self.main_dialog.clustered_locs,
                oversampling=optimal_oversampling,
                viewport=self.viewport,
                blur_method="convolve",
            )

            # scale both images
            image_locs = self.main_dialog.window.view.scale_contrast(
                image_locs, autoscale=True
            )
            image_clustered = self.main_dialog.window.view.scale_contrast(
                image_clustered, autoscale=True
            )       

            # get colors for all locs and clusters (latter will be white anyway)
            colors = [[0, 1, 1], [1, 0, 0]]

            # create bgra
            Y, X = image_locs.shape
            bgra = np.zeros((Y, X, 4), dtype=np.float32)

            # fill bgra with colored images
            for color, im in zip(colors, [image_locs, image_clustered]):
                bgra[..., 0] += color[2] * im
                bgra[..., 1] += color[1] * im
                bgra[..., 2] += color[0] * im
            bgra = np.minimum(bgra, 1)

            # convert bgra to 8 bit
            bgra = self.main_dialog.window.view.to_8bit(bgra)       
            bgra[..., 3].fill(255)

            # make it a qimage
            qimage = QtGui.QImage(
                bgra.data, X, Y, QtGui.QImage.Format_RGB32
            )

            # scale to widget
            qimage = qimage.scaled(
                self.width(),
                self.height(),
                QtCore.Qt.KeepAspectRatioByExpanding,
            )
            # draw cluster success, barcode names
            if self.main_dialog.show_class_name.isChecked():
                qimage = self.draw_cluster_names(qimage)
            if self.main_dialog.show_cluster_success.isChecked():
                qimage = self.draw_cluster_success(qimage)

            # set pixmap
            self.pixmap = QtGui.QPixmap.fromImage(qimage)
            self.setPixmap(self.pixmap)

    def draw_cluster_names(self, image):
        image = image.copy()
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QColor("red"))

        # get the right font size
        font = painter.font()
        d_display, _ = self.map_to_view(
            (self.main_dialog.pick_diameter, 0)
        )
        font.setPixelSize(d_display / 4.5)
        painter.setFont(font)

        # find offset to display in the top right corner
        offset_x = d_display / 3
        offset_y = -d_display / 3

        # display the names of classified barcodes
        for i, pos in enumerate(self.main_dialog.centers):
            name = str(self.main_dialog.test_names[i])
            cx, cy = self.map_to_view(pos)
            painter.drawText(
                cx + offset_x, cy + offset_y, name
            )

        painter.end()
        return image

    def draw_cluster_success(self, image):
        image = image.copy()
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QColor("red"))

        # get the right font size
        font = painter.font()
        d_display, _ = self.map_to_view(
            (self.main_dialog.pick_diameter, 0)
        ) # pick diameter in display units
        font.setPixelSize(d_display / 4.5)
        painter.setFont(font)

        # find offset to display in the top right corner, below names
        offset_x = d_display / 3
        offset_y = -d_display / 3 + 0.3 + d_display / 4.5

        #display the first two decimals of the actual success
        for i, pos in enumerate(self.main_dialog.centers):
            success = np.round(self.main_dialog.cluster_success[i], 2)
            cx, cy = self.map_to_view(pos)
            painter.drawText(
                cx + offset_x, cy + offset_y, str(success)
            )

        painter.end()
        return image


class Plugin():
    def __init__(self, window):
        self.name = "render"
        self.window = window

    def execute(self):
        pp_menu = self.window.menus[3] # postprocessing menu
        pp_menu.addSeparator()

        _action = pp_menu.addAction(
            "Perform barcode classification"
        )
        _action.triggered.connect(self.open_dialog)

        # add dialog
        self.window._dialog = MainDialog(self.window)
        self.window.dialogs.append(self.window._dialog)

    def open_dialog(self):
        if len(self.window.view.locs) != 1:
            message = (
                "Please load only one localizations file."
            )
            QtWidgets.QMessageBox.information(self.window, "Warning", message)
        else:
            self.window._dialog.show()
