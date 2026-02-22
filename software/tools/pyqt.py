

from tools.nidevice import NiDevice
from tools.camera import BaslerCam
from tools.model import Model


from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSlot, QCoreApplication
from PyQt6.QtGui import QImage, QPixmap

from PyQt6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QMainWindow
)
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from tools.tools import *

import json
from pathlib import Path
from datetime import datetime

class MainWindow(QMainWindow):

    send_camera = pyqtSignal(object)
    send_ni = pyqtSignal(object)
    send_worker = pyqtSignal(object)
    send_model = pyqtSignal(object)

    def __init__(self, args):
        super().__init__()

        self.setWindowTitle("Micromanipulator Controller V2.0")

        # self.width = 1020
        # self.height = 720
        # self.setGeometry(0, 0, self.width, self.height)

        self.params = {}
        self.args = args

        self.livestream_flag = False
        self.calibration_flag = False
        self.measurement_flag = False

        self.init_UI()
        
        self.init_driver()
        self.init_camera()
        self.init_model()

        # Update window title to show camera status
        self.update_window_title()

        self.showMaximized()


    def init_UI(self):

        self.win = QWidget()
        # self.win.resize(self.width,self.height)

        self.main_layout = QVBoxLayout()
        self.top_row = TopRow(self)
        self.main_layout.addWidget(self.top_row)
           
        self.tabs = QTabWidget()

        self.main_tab = MainTab(self)
        self.tabs.addTab(self.main_tab, "Main")

        self.config_tab = ConfigTab(self)
        self.tabs.addTab(self.config_tab, "Config")

        self.analysis_tab = AnalysisTab(self)
        self.tabs.addTab(self.analysis_tab, "Analysis")

        self.main_layout.addWidget(self.tabs)

        self.win.setLayout(self.main_layout)
        self.setCentralWidget(self.win) 


    def init_driver(self):

        #Init driver and signal pipe
        if not hasattr(self, 'driver'):
            self.driver = NiDevice(self.args)
            if not self.driver.connected:
                return -1

            self.driver.NI_setData.connect(self.receive_NI_data) #signals
            self.driver.NI_print_str.connect(self.receive_NI_str) #signals
            self.send_ni.connect(self.driver.receive_main)

        if not hasattr(self, 'ni_process'):
            self.ni_process = QtCore.QThread()
            self.driver.moveToThread(self.ni_process)
            self.ni_process.started.connect(self.driver.run)
        elif self.ni_process.isRunning():
            # Thread is still running, wait for it to finish
            self.ni_process.quit()
            self.ni_process.wait(3000)

    def init_camera(self):

        if not hasattr(self, 'cam'):
            # Enable emulation environment variables before creating camera
            import os
            if not os.environ.get('PYLON_CAMEMU'):
                os.environ['PYLON_CAMEMU'] = '1'
                os.environ['PYLON_CAMEMU_COUNT'] = '1'
            
            self.cam = BaslerCam(self.args)

            if not self.cam.connected:
                return -1
            
                
            self.cam.send_data.connect(self.receive_cam) #signals
            self.send_camera.connect(self.cam.receive_main)
            self.cam.send_image.connect(self.receive_image)
            self.cam.basler_print_str.connect(self.receive_cam_str) #signals

        if not hasattr(self, 'cam_process'):
            self.cam_process = QtCore.QThread()
            self.cam.moveToThread(self.cam_process)
            self.cam_process.started.connect(self.cam.run)
        elif self.cam_process.isRunning():
            # Thread is still running, wait for it to finish
            self.cam_process.quit()
            self.cam_process.wait(3000)

    def init_model(self):

        #Connect model
        if not hasattr(self, 'model'):
            self.model = Model(self.args)
            if not self.model.connected:
                return -1
            
            self.model.send_main.connect(self.receive_model)
            self.model.send_model_str.connect(self.receive_model_str)
            self.send_model.connect(self.model.receive_main)

            self.model_process = QtCore.QThread()
            self.model.moveToThread(self.model_process)
            self.model_process.started.connect(self.model.run)

        elif self.model_process.isRunning():
            # Thread is still running, wait for it to finish
            self.model_process.quit()
            self.model_process.wait(3000)

    def update_params(self):
        
        """Update parameters from UI to threads"""
        # Update NI driver params
        self.params["timestamp_sync"] = float(time.time())
        self.params["res_1"] = float(self.main_tab.sliderR1.value()/100)
        #camera params
        self.params["trackFlag"] = bool(self.config_tab.tracker_box.isChecked())
        self.params["Flag_save"] = bool(self.config_tab.saving_box.isChecked())
        self.params["manual_input"] = bool(self.config_tab.manual_box.isChecked())

        # Get values from QLineEdit objects (use .text())
        for i in self.config_tab.param:
            self.params[i] = float(self.config_tab.param[i].text())
        
        # Get values from QComboBox objects (use .currentText())
        self.params["Flag_feedBack"] = str(self.config_tab.combobox_feedback.currentText())
        self.params["Flag_mode"] = str(self.config_tab.combobox_test.currentText())
        self.params["current_sequence"] = str(self.config_tab.combobox_wave.currentText())

        # Update model params

    @pyqtSlot(str)
    def receive_NI_str(self, msg):
        if "Calibration completed" in msg:
            self.calibration_flag = False
            print("Setting calibration flag to false")

        self.main_tab.update_str_field(msg)

    @pyqtSlot(str)
    def receive_cam_str(self, msg):
        self.main_tab.update_str_field(msg)

    @pyqtSlot(str)
    def receive_model_str(self, msg):
        self.main_tab.update_str_field(msg)

    @pyqtSlot(object)
    def receive_model(self, msg):
        self.send_ni.emit(Task(priority=999, id = "1003", data=msg))

    @pyqtSlot(object)
    def receive_NI_data(self, data):
        # print("receiving NI data", data)
        self.main_tab.plots_tab["currents"].update_plot(data["timestamp"], data["current_target"], "I1_target")
        self.main_tab.plots_tab["currents"].update_plot(data["timestamp"], data["current_measured"], "I1_measured")
        self.main_tab.plots_tab["magnetic_field"].update_plot(data["timestamp"], data["magnetic_field"], "B1_measured")

        self.main_tab.CurrentMeasuredLabel.setText("Measured Current: {:.2f} A".format(data["current_measured"]))
        self.main_tab.CurrentTargetLabel.setText("Target Current: {:.2f} A".format(data["current_target"]))
        self.main_tab.BMeasuredLabel.setText("Measured B: {:.2f} []".format(data["magnetic_field"]))

    @pyqtSlot(object)
    def receive_cam(self, data):
        self.main_tab.receiveTrackData(data)
        

    @pyqtSlot(object)
    def receive_image(self, data):
        self.main_tab.setImage(data)
            
    def stop_all_threads(self):
        """Stop all worker threads safely and quickly"""
        
        
        # Send stop signals AFTER setting flags
        stop_task = Task(priority=999, id="1000", data=None)
        
        if self.ni_process.isRunning():#hasattr(self, 'driver'):
            self.send_ni.emit(stop_task)
            print("DEBUG: Stop signal sent to NI device")
        
        if self.cam_process.isRunning():#hasattr(self, 'cam'):
            self.cam._running = False
            self.send_camera.emit(stop_task)
            print("DEBUG: Stop signal sent to camera")
            
        if self.model_process.isRunning():#hasattr(self, 'model'):
            self.send_model.emit(stop_task)
            print("DEBUG: Stop signal sent to model")
        
        QCoreApplication.processEvents()
        time.sleep(0.1)  # Small delay to allow workers to process stop commands
        # Reset UI flags
        self.livestream_flag = False
        self.calibration_flag = False  
        self.measurement_flag = False

        self.cam.mode = "0"
        self.driver.mode = "0"
        
        # Check which threads need to be waited for
        threads_to_wait = []
        if self.ni_process.isRunning():
            self.driver._running = False
            threads_to_wait.append(('NI', self.ni_process))
            print("DEBUG: NI thread still running, will wait...")
        if self.cam_process.isRunning():
            self.cam._running = False
            threads_to_wait.append(('Camera', self.cam_process))
            print("DEBUG: Camera thread still running, will wait...")
        if self.model_process.isRunning():
            self.model._running = False
            threads_to_wait.append(('Model', self.model_process))
            print("DEBUG: Model thread still running, will wait...")
    
        # Wait for threads with progressive timeout
        for name, process in threads_to_wait:
            if not process.wait(100):  # First try 1 second
                process.quit()
                if not process.wait(100):  # Give another second after quit
                    print(f"DEBUG: {name} thread didn't not close, terminating...")
                    process.terminate()

        # Clear plot data first (quick operation)
        QCoreApplication.processEvents()
        self.main_tab.clear_plot_data()
    
    def closeEvent(self, event):
        """Handle application close event"""
        # First stop ongoing operations
        self.stop_all_threads()

        # Driver has nothing to close here
        if hasattr(self, 'driver'):
            pass

        if self.cam.connected:
            self.cam.cam.Close()

        # Driver has nothing to close here
        if hasattr(self, 'model'):
            pass

        event.accept()
        QtWidgets.QApplication.quit()
        
    def stopEvent(self):
        """Handle application close event"""
        self.stop_all_threads()
        QCoreApplication.processEvents()
        self.reset_all_event()

        self.main_tab.update_str_field("Application stopped by user.")
    
    def stop_calibration(self):
        """Stop only the calibration process"""
        if self.calibration_flag:
            # Send stop signal to NI device to stop current operation
            stop_task = Task(priority=999, id="1000", data=None)
            self.send_ni.emit(stop_task)

            if self.ni_process.isRunning():
                self.driver._running = False
                time.sleep(0.1)  # Small delay to allow worker to process stop command
                if not self.ni_process.wait(100):  # First try 1 second
                    self.ni_process.quit()
                    if not self.ni_process.wait(100):  # Give another second after quit
                        self.ni_process.terminate()

            # Reset only calibration flag
            self.calibration_flag = False
            QCoreApplication.processEvents()
            self.reset_all_event()
            self.main_tab.update_str_field("Calibration stopped by user.")
            self.main_tab.clear_plot_data()

    def stop_livestream(self):
        """Stop only the livestream process"""
        if self.livestream_flag:
            # Just set the camera _running flag to stop current livestream
            stop_task = Task(priority=999, id="1000", data=None)
            self.send_camera.emit(stop_task)

            if self.cam_process.isRunning():
                self.cam._running = False
                time.sleep(0.1)  # Small delay to allow worker to process stop command
                if not self.cam_process.wait(100):  # First try 1 second
                    self.cam_process.quit()
                    if not self.cam_process.wait(100):  # Give another second after quit
                        self.cam_process.terminate()
    
            # Reset only livestream flag
            self.livestream_flag = False
            QCoreApplication.processEvents()

            self.reset_all_event()
            self.main_tab.update_str_field("Livestream stopped by user.")
            self.main_tab.clear_plot_data()
    
    def stop_measurement(self):
        """Stop only the measurement process"""
        if hasattr(self, 'driver') and hasattr(self, 'cam') and self.measurement_flag:
            # Send stop signals to both driver and camera
            stop_task = Task(priority=999, id="1000", data=None)
            self.send_ni.emit(stop_task)
            self.send_camera.emit(stop_task)
            
            # Process events to ensure stop signals are delivered
            QCoreApplication.processEvents()
            
            # Send a flush signal to clear any remaining tasks
            flush_task = Task(priority=1000, id="flush", data=None)
            self.send_ni.emit(flush_task)
            self.send_camera.emit(flush_task)
            
            # Also send flush to model if it exists
            if hasattr(self, 'model'):
                self.send_model.emit(flush_task)
            
            # Send flush to tracker through camera's frame_for_tracker signal
            if hasattr(self, 'cam') and hasattr(self.cam, 'frame_for_tracker'):
                self.cam.frame_for_tracker.emit(flush_task)
            
            # Small delay to allow workers to process stop and flush commands
            import time
            time.sleep(0.1)
            
            # Process events again to clear any remaining signals
            QCoreApplication.processEvents()
            
            # Reset only measurement flag
            self.measurement_flag = False

    def reset_all_event(self):

        self.snapimage_flag = False
        self.livestream_flag = False
        self.calibration_flag = False
        self.measurement_flag = False

        self.model._reset_variables()
        self.driver._reset_variables()
        self.cam._reset_variables()

        self.cam.mode = "0"
        self.driver.mode = "0"

            
    def start_calibration(self):
        """Start calibration without recreating threads"""
        if not hasattr(self, 'driver') or not self.driver.connected:
            print("Driver not connected")
            return False
            
        self.update_params()
        # Force voltage feedback mode for calibration
        self.params["Flag_feedBack"] = "voltage"
        
        self.driver._update_params(self.params)
        self.driver.mode = "1"
        self.calibration_flag = True
        
        if not self.ni_process.isRunning():
            self.ni_process.start()

        return True
        
    def start_livestream(self):
        """Start livestream without recreating threads"""
        if not hasattr(self, 'cam') or not self.cam.connected:
            print("Camera not connected")
            return False
            
        self.update_params()
        self.cam._update_params(self.params)
        
        # Reset camera running flag for new session
        self.cam._running = True
        self.livestream_flag = True
        self.cam.mode = "1"
        
        if not self.cam_process.isRunning():
            self.cam_process.start()

        return True
        
    def start_measurement(self):

        """Start measurement without recreating threads"""
        if not hasattr(self, 'driver') or not self.driver.connected:
            print("Driver not connected")
            return False
        if not hasattr(self, 'cam') or not self.cam.connected:
            print("Camera not connected") 
            return False
        
        # Reset worker states for new session
        # Clear any pending stop tasks first by resetting worker variables
        self.driver._reset_variables()
        self.cam._reset_variables()
        
        # Process any pending Qt events to clear old signals
        QCoreApplication.processEvents()

        test_mode = self.config_tab.combobox_test.currentText()
        self.update_params()

        self.driver._update_params(self.params)
        self.cam._update_params(self.params)

        if test_mode == "position":
            if hasattr(self, 'model') and self.model.connected and (len(self.main_tab.boundaryFinal) == 2):
                # Set model type BEFORE loading model data
                if self.config_tab.emp_box.isChecked():
                    self.model.model_type = "EMP"
                else:
                    self.model.model_type = "FEM"

                # Load model data according to selected type
                self.model._update_params(self.params)

                self.cam.finalboundaries = self.main_tab.boundaryFinal
                self.cam.trackFlag = True
                self.cam._init_tracker()

                self.model._init_params(np.abs((self.main_tab.x2+self.main_tab.x1)/2), np.abs(((self.main_tab.y2+self.main_tab.y1)/2)), self.params["offset"])

                if not self.model_process.isRunning():
                    self.model_process.start()
            else:
                return False
        elif self.config_tab.tracker_box.isChecked():
            if len(self.main_tab.boundaryFinal) == 2:
                self.cam.finalboundaries = self.main_tab.boundaryFinal
                self.cam.trackFlag = True
                self.cam._init_tracker()
            else:
                return False
        
        
        if self.config_tab.manual_box.isChecked():
            self.driver.Flag_mode = "manual"
        
        # Small delay to ensure all pending signals are processed
        import time
        time.sleep(0.1)
        
        # Now set running states and modes
        self.driver._running = True
        self.cam._running = True
        
        # Set modes
        self.driver.mode = "2"  # measurement mode
        self.cam.mode = "2"     # recording mode
        # Start threads if not running


        if not self.ni_process.isRunning():
            self.ni_process.start()
            
        if not self.cam_process.isRunning():
            self.cam_process.start()
        
        self.measurement_flag = True

        if self.params["Flag_save"]:
            self.save_params_meta()

        return True
    
    def update_window_title(self):
        """Update window title to show camera and device status"""
        base_title = "Micromanipulator Controller V2.0"
        
        status_parts = []
        
        # Camera status
        if hasattr(self, 'cam') and self.cam.connected:
            if getattr(self.cam, 'is_emulated', False):
                status_parts.append("Camera: EMULATED")
            else:
                status_parts.append("Camera: REAL")
        else:
            status_parts.append("Camera: DISCONNECTED")
        
        # Driver status  
        if hasattr(self, 'driver') and self.driver.connected:
            status_parts.append("NI: CONNECTED")
        else:
            status_parts.append("NI: DISCONNECTED")
        
        if status_parts:
            title = f"{base_title} - {' | '.join(status_parts)}"
        else:
            title = base_title
            
        self.setWindowTitle(title)

    def save_params_meta(self): 

        # Merge optional extra info
        payload = dict(self.params)

        # Basic run info
        payload.setdefault("_meta", {})
        payload["_meta"]["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Serialize numpy types safely
        def _ser(x):
            try:
                import numpy as np
                if isinstance(x, (np.generic,)):
                    return x.item()
            except Exception:
                pass
            return x

        payload = {k: _ser(v) for k, v in payload.items()}

        out =  Path(f"./results/meta.json")
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
