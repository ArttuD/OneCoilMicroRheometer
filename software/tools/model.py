"""
Simulation Model for Microrheology Controller  
"""

import logging
import queue
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import scipy.ndimage
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QObject, QCoreApplication

from tools.config import get_config
from tools.config import get_config
import joblib
import os

import h5py
import csv

from tools.tools import LatestQueue

class Model(QObject):
    """
    Magnetic field simulation model for microrheology experiments.
    """

    # Qt Signals
    send_main = pyqtSignal(object)
    send_model_str = pyqtSignal(str)

    def __init__(self, args: Any, parent: Optional[QObject] = None) -> None:
        """
        Initialize the simulation model.
        """
        super().__init__(parent)
        
        self.args = args
        self.parent = parent

        try:
            # Load simulation parameters from configuration
            config = get_config()
            model_params = config.get_model_params()
            self.root = model_params['simulation_data_root']
            self.filter_sigma = model_params['simulation_filter_sigma']
        except Exception as e:
            self.send_model_str.emit(f"Failed to load model configuration: {e}")
            raise
    
        # Load camera parameters from configuration
        config = get_config()
        cam_params = config.get_camera_params()
        
        pixel_size = cam_params['pixel_size']
        objective = cam_params['objective']
        light_path_zoom = cam_params['light_path_zoom']
        camera_adapter_zoom = cam_params['camera_adapter_zoom']
        camera_height = cam_params['camera_height']
        camera_width = cam_params['camera_width']
        
        # Calculate optical system parameters
        self.m = pixel_size / (objective * light_path_zoom * camera_adapter_zoom) #* 1e-6
        self.height = camera_height * self.m * 1e-6
        self.width = camera_width * self.m * 1e-6
        self.center = self.height / 2

        self.data_queue = LatestQueue()
        self.pre_init = False
        self.model_type = "EMP"

        self._init_variables()
        self.connected = True

    def _init_variables(self) -> None:

        """Initialize instance variables to default states."""
        self.force_matrix: Optional[np.ndarray] = None
        self.force_matrix_gradient: Optional[np.ndarray] = None

        self.x_ticks: Optional[np.ndarray] = None  
        self.y_ticks: Optional[np.ndarray] = None
        self.i_ticks: Optional[np.ndarray] = None

        self._running: bool = False
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.i: Optional[float] = None
        
        # State tracking
        self.force: float = 0.0
        self.force_to_current: float = 1.0 

    def _reset_variables(self) -> None:
        """Reset model state and clear queues"""
        
        self._running = False

        self.i = None
        self.x = None
        self.y = None
        
        # Clear the data queue
        self.data_queue.clear()
        
        # Reinitialize with fresh queue
        self.data_queue = LatestQueue()
        
        # Reset state tracking
        self.force: float = 0.0
        self.force_to_current: float = 1.0 
        
    def _update_params(self, param):

        # Load camera parameters from configuration
        config = get_config()
        cam_params = config.get_camera_params()
        
        self.pixel_size = cam_params['pixel_size']
        self.objective = cam_params['objective']
        self.light_path_zoom = cam_params['light_path_zoom']
        self.camera_adapter_zoom = cam_params['camera_adapter_zoom']
        self.camera_height = cam_params['camera_height']
        self.camera_width = cam_params['camera_width']
        
        #param["m"] 
        self.sync_time = param["timestamp_sync"]
        self.m = self.pixel_size/(self.objective*self.light_path_zoom*self.camera_adapter_zoom)#*1e-6
        self.height = param["height"] = self.camera_height*self.m*1e-6
        self.width = param["width"] = self.camera_width*self.m*1e-6
        self.start_w = param["start_w"] =  0.0035 - self.width
        self.end_w = param["end_w"] = 0.0035
        self.start_h = param["start_h"] = 0.003 - self.height/2
        self.end_h = param["end_h"] = 0.003 + self.height/2
        self.offset = param["offset"] 

        if self.model_type == "FEM":
            self.force_matrix, self.force_matrix_gradient, self.x_ticks, self.y_ticks, self.i_ticks = self.load_csv()
        elif self.model_type == "EMP":
            self.krr_bundle, self.krr_bundle_ref  = self.load_models()

    def load_models(self):
        """Load empirical model bundles with version-robust fallbacks.
        Implement for force corrected single model 
        """
        bundle_krr = None
        # bundle_gpr = None

        bundle_krr = joblib.load(os.path.join("./Analysis/results", "force_krr.joblib")) #Dataset 1
        krr_bundle_ref = joblib.load(os.path.join("./Analysis/results", "force_krr_ref.joblib")) #Dataset 2
        # bundle_gpr = joblib.load(os.path.join("./configs", "force_gpr.joblib"))

        return bundle_krr, krr_bundle_ref


    def load_csv(self):

        try:
            with h5py.File("./Analysis/results/force_model.h5", "r") as f:
                force_control= f["model/force_field"][...]
                force_matrix_gradient = f["model/F2I"][...]
                x_axis = f["model/x"][...]
                y_axis = f["model/y"][...]
                i_axis = f["model/i"][...]

            return force_control, force_matrix_gradient, x_axis, y_axis, i_axis
        
        except Exception as e:
            self.send_model_str.emit(f"Failed to load CSV data: {e}")
            raise
    
    def _init_params(self, x, y, i):

        self.x = x*self.m*1e-6
        self.y = (y)*self.m*1e-6 - 3*self.center
        self.i = self.offset

        if self.model_type == "FEM":
            x_idx = np.argmin(np.abs((self.x_ticks-np.round(self.x,5))))
            y_idx = np.argmin(np.abs((self.y_ticks-np.round(self.y,5))))
            i_idx = np.argmin(np.abs((self.i_ticks-np.round(self.i,5))))
            self.force = self.force_matrix[y_idx, x_idx, i_idx]
            self.force_to_current = self.force_matrix_gradient[y_idx, x_idx, i_idx]
            self.pre_init = True

        elif self.model_type == "EMP":
            # Prefer KRR if available, otherwise GPR; if neither, default to zero and warn.
            # Would be better to make 1 predictor model for speed: current is first version... happened to measure with that
            self.force = self.predict(self.krr_bundle, [[self.x, self.y, self.i]]) - self.predict(self.krr_bundle_ref, [[self.x, self.y, self.i]])
            self.pre_init = True

        # Ensure results directory exists and always start with a fresh file
        out_path = "./results/model_output.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter=",").writerow([self.x, self.y, self.i, 0, self.force, 0, self.force_to_current, 0, 0.0])
    

    def predict(self, bundle, X_new):
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        # Ensure column count matches feature_order
        if X_new.shape[1] != len(bundle["feature_order"]):
            raise ValueError(f"Expected {len(bundle['feature_order'])} features: {bundle['feature_order']}")

        Xs = bundle["sx"].transform(X_new)
        y_pred_scaled = bundle["model"].predict(Xs)
        # sy may be MinMaxScaler or StandardScaler; inverse_transform expects 2D
        y_pred = bundle["sy"].inverse_transform(np.asarray(y_pred_scaled).reshape(-1, 1)).ravel()
        return y_pred[0]
    
    def _find_lin_bounds(self, ticks, v):

        n = len(ticks)
        if n < 2:
            return 0, 0, 0.0
        j = int(np.searchsorted(ticks, v))
        i0 = max(0, min(n - 2, j - 1))
        i1 = i0 + 1
        t0 = float(ticks[i0])
        t1 = float(ticks[i1])

        if t1 == t0:
            w = 0.0
        else:
            w = (float(v) - t0) / (t1 - t0)
            if w < 0.0:
                w = 0.0
            elif w > 1.0:
                w = 1.0

        return i0, i1, w

    def _trilinear(self, vol, x_ticks, y_ticks, i_ticks,x, y, i):

        y0, y1, wy = self._find_lin_bounds(y_ticks, y)
        x0, x1, wx = self._find_lin_bounds(x_ticks, x)
        i0, i1, wi = self._find_lin_bounds(i_ticks, i)

        # Corner values
        V000 = float(vol[y0, x0, i0])
        V100 = float(vol[y0, x1, i0])
        V010 = float(vol[y1, x0, i0])
        V110 = float(vol[y1, x1, i0])
        V001 = float(vol[y0, x0, i1])
        V101 = float(vol[y0, x1, i1])
        V011 = float(vol[y1, x0, i1])
        V111 = float(vol[y1, x1, i1])

        # Interpolate
        c00 = V000 * (1 - wx) + V100 * wx
        c10 = V010 * (1 - wx) + V110 * wx
        c01 = V001 * (1 - wx) + V101 * wx
        c11 = V011 * (1 - wx) + V111 * wx

        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy

        val = c0 * (1 - wi) + c1 * wi
        return val
    
    def findPoint(self, x, y, i) -> None:
        """
        Calculate magnetic field at specified position and emit results.
        """
        # try:
            # Convert coordinates to simulation space
        y_cand = y * 1e-6 * self.m - self.center 
        x_cand = x * 1e-6 * self.m - 1e-4 #position compensation based on coil position 
        #dataset 1 -> 1e-4, dataset 1 -> 1.5e-4. SHould make this a config parameter but laze
        i_cand = i

        try:
            f_current = self._trilinear(self.force_matrix, self.x_ticks, self.y_ticks, self.i_ticks,
                                        x_cand, y_cand, i_cand)
            force_to_current = self._trilinear(self.force_matrix_gradient, self.x_ticks, self.y_ticks, self.i_ticks,
                                               x_cand, y_cand, i_cand)
        except Exception:
            # Nearest-neighbor fallback
            x_idx = int(np.argmin(np.abs((self.x_ticks - x_cand))))
            y_idx = int(np.argmin(np.abs((self.y_ticks - y_cand))))
            i_idx = int(np.argmin(np.abs((self.i_ticks - i_cand))))
            f_current = float(self.force_matrix[y_idx, x_idx, i_idx])
            force_to_current = float(self.force_matrix_gradient[y_idx, x_idx, i_idx])

        if self.pre_init == True:

            #dummy fix for first run
            self.force = f_current
            self.pre_init = False

        # Calculate error from previous position
        error = float(np.abs(self.force - f_current))

        # self.past = f_current
        # Convert to current reference
        d_i_ref = error / force_to_current if force_to_current != 0 else 0.0

        d_i_ref = d_i_ref if i_cand-d_i_ref >= 0 else 0

        # Prepare data packet
        # out_path = "./results/model_output.csv"
        if i == 0:
            d_i_ref = 0
        # Prepare data packet
        with open("./debug.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter=",").writerow([x_cand, y_cand, i_cand, error, self.force, f_current, force_to_current, d_i_ref, time.time() - self.sync_time])
        
        data = {
            "x": float(x_cand),
            "y": float(y_cand),
            "force": float(f_current),
            "force_to_current": float(force_to_current),
            "current": float(i_cand),
            "d_i_ref": float(d_i_ref),
            "timestamp": time.time() - self.sync_time
        }

        # print("Model sending queue size, data:", self.data_queue.qsize(), data["timestamp"])

        self.send_main.emit(data)
        # print("sending data:", data["timestamp"])
        # except Exception as e:
        #     self.send_model_str.emit(f"Error in findPoint: {e}")

    def CompPoint(self, x, y, i) -> None:
        """
        Calculate magnetic field at specified position and emit results.
        """
        # try:
            # Convert coordinates to simulation space
        y_cand = y * 1e-6 * self.m - self.center 
        x_cand = x * 1e-6 * self.m
        i_cand = i

        # Prefer KRR if available, otherwise GPR; if neither, compute zero
        f_current = self.predict(self.krr_bundle, [[x_cand, y_cand, i_cand]])
        f_ref_current = self.predict(self.krr_bundle_ref, [[x_cand, y_cand, i_cand]])

        if self.pre_init == True:
            #dummy fix for first run
            self.force = f_current - f_ref_current
            self.pre_init = False

        # Numerical derivative w.r.t current using same bundle if available
        f_prev = self.predict(self.krr_bundle, [[x_cand, y_cand, i_cand - 0.2]])
        f_ref_prev = self.predict(self.krr_bundle_ref, [[x_cand, y_cand, i_cand - 0.2]])

        force_to_current = ((f_current - f_ref_current) - (f_prev-f_ref_prev)) / 0.2

        # Calculate error from previous position
        error = np.abs(self.force - (f_current - f_ref_current))
        # self.past = f_current
        # Convert to current reference
        d_i_ref = error / force_to_current

        d_i_ref = d_i_ref if i_cand-d_i_ref >= 0 else 0

        # Prepare data packet
        with open("./debug.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter=",").writerow([x_cand, y_cand, i_cand, error, self.force, f_current, force_to_current, d_i_ref, time.time() - self.sync_time])
        
        data = {
            "x": x_cand,
            "y": y_cand,
            "force": float(f_current),
            "force_to_current": float(force_to_current),
            "current": float(i_cand),
            "d_i_ref": float(d_i_ref),
            "timestamp": time.time() - self.sync_time
        }

        if i == 0:
            d_i_ref = 0

        self.send_main.emit(data)
        
    @pyqtSlot(object)
    def receive_main(self, param):
        if param.id == "1000":
            self._running = False
        elif param.id == "flush":
            print("Model received flush command - clearing queue")
            # Clear the data queue
            self.data_queue.clear()

        elif param.id == "1001":
            self.data_queue.put(param)
        elif param.id == "1002":
            self.data_queue.put(param)
        
    def run(self):
        try:
            self._running = True
            i = 0
            while self._running:
                param = self.data_queue.get_latest(timeout=0.01)
                if param is not None:
                    try:
                        if param.data[0] == -1:
                            self.send_main.emit({
                                "x": None,
                                "y": None,
                                "force": 0.0,
                                "force_to_current": 0.0,
                                "current": 0.0,
                                "d_i_ref": 0.0,
                                "timestamp": time.time() - self.sync_time
                            })
                        else:
                            if self.model_type == "FEM":
                                self.findPoint(param.data[0], param.data[1],param.data[2])
                            elif self.model_type == "EMP":
                                self.CompPoint(param.data[0], param.data[1],param.data[2])
                    except Exception as e:
                        self.send_model_str.emit(f"Model processing error: {e}")
                else:
                    time.sleep(0.001)
                    QCoreApplication.processEvents()

        except Exception as e:
            self.send_model_str.emit(f"Model error: {e}")
            # time.sleep(0.1)

        self.pre_init = False
        QCoreApplication.processEvents()


