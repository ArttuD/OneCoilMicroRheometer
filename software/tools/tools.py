import time

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLineEdit, QGraphicsRectItem, QWidget, QLabel, QSlider, QPushButton, QGridLayout, QVBoxLayout, QComboBox, QCheckBox
from pyparsing import Optional
import pyqtgraph as pg
from PyQt6.QtGui import QFont, QPen, QBrush, QImage, QPixmap
import numpy as np
from tools.config import get_config
from typing import Dict, Optional, Tuple, Any, Union

import threading
from collections import deque


# Load configuration parameters from centralized config
config = get_config()
colors_list = config.get_colors_list()
multi_choices = config.get_multi_choices()
plot_dict = config.get_plot_dict()


class LatestQueue:
    def __init__(self) -> None:
        self._dq = deque(maxlen=1)
        self._event = threading.Event()
        self._lock = threading.Lock()

    def put(self, item: Any) -> None:
        with self._lock:
            self._dq.append(item)
            self._event.set()

    def get_latest(self, timeout: Optional[float] = None) -> Optional[Any]:
        if not self._event.wait(timeout):
            return None
        with self._lock:
            if not self._dq:
                self._event.clear()
                return None
            item = self._dq[-1]
            self._dq.clear()
            self._event.clear()
            return item

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()
            self._event.clear()

    def empty(self) -> bool:
        with self._lock:
            return len(self._dq) == 0

    def qsize(self) -> int:
        with self._lock:
            return len(self._dq)
        
class Task:
    def __init__(self, priority, id, data, *args, **kwargs):
        self.priority = priority
        self.id = id
        self.data = data
        self.args = args
        self.kwargs = kwargs

    def __lt__(self, other):
        return self.priority < other.priority
    
class TopRow(QWidget):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent  # Reference to MainWindow for accessing shared data/methods

        layout = QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        #Start measurement
        self.btnStart = QPushButton("Start measurement")
        self.btnStart.pressed.connect(self.start)
        self.btnStart.setStyleSheet("background-color : green")

        #Stop measurement
        self.btnStop = QPushButton("Stop measurement")
        self.btnStop.pressed.connect(self.stop)
        self.btnStop.setStyleSheet("background-color : red")

        #Stop measurement
        self.btnCalib = QPushButton("Calibrate")
        self.btnCalib.pressed.connect(self.calibrate)
        # self.btnCalib.setStyleSheet("background-color : blue")

        self.btnSnap = QPushButton("Snap Image")
        self.btnSnap.pressed.connect(self.snap_image)
        # self.btnSnap.setStyleSheet("background-color : blue")

        self.streamBtn = QPushButton("Live")
        self.streamBtn.pressed.connect(self.livestream)
        # self.streamBtn.setStyleSheet("background-color : green")
        
        self.clearBtn = QPushButton("Clear plots")
        self.clearBtn.pressed.connect(self.clear_plots)
        # self.streamBtn.setStyleSheet("background-color : green")


        self.btnExit = QPushButton("Exit")
        self.btnExit.pressed.connect(self.exit)
        # self.btnExit.setStyleSheet("background-color : blue")

        layout.addWidget(self.btnStart, 0, 0)
        layout.addWidget(self.btnStop, 0, 1)
        layout.addWidget(self.btnCalib, 0, 2)
        layout.addWidget(self.btnSnap, 0, 3)
        layout.addWidget(self.streamBtn, 0, 4)
        layout.addWidget(self.clearBtn, 0, 5)
        layout.addWidget(self.btnExit, 0, 6)
        layout.addWidget(self.btnExit, 0, 6)
        
        # Add a status label that spans all columns in row 1
        layout.addWidget(QLabel("Messages:"), 1, 0)
        self.printfield = QLineEdit()
        self.printfield.setReadOnly(True)
        self.printfield.setFixedHeight(25)

        layout.addWidget(self.printfield, 2, 0, 1, 6)  # row=1, col=0, rowspan=1, colspan=6

        self.setLayout(layout)

    def start(self):
        print("Measurement started")
        success = self.parent.start_measurement()
        if not success:
            print("Failed to start measurement")
            return -1

    def calibrate(self):
        
        if not self.parent.calibration_flag:
            success = self.parent.start_calibration()
            if not success:
                print("Failed to start calibration")
                self.printfield.setText("Failed to start calibration")
        else:
            self.parent.stop_calibration()
            self.printfield.setText("Calibration stopped")


    def livestream(self):

        if not self.parent.livestream_flag:
            success = self.parent.start_livestream()
            if not success:
                print("Failed to start livestream")
        else:
            self.parent.stop_livestream()

    def snap_image(self):
        if self.parent.cam.connected:
            self.parent.update_params()
            self.parent.cam._update_params(self.parent.params)
            self.parent.cam.snapImage()  # Fixed method name
            self.parent.main_tab.video_widget.clear_overlays()
        

    def stop(self):
        # Smart stop - figure out what's running and stop it appropriately
        if self.parent.measurement_flag:
            # self.parent.stop_measurement()
            self.parent.stopEvent()
        elif self.parent.calibration_flag:
            self.parent.stop_calibration()
        elif self.parent.livestream_flag:
            self.parent.stop_livestream()
        else:
            # Nothing specific running, use general stop
            self.parent.stopEvent()
    
    def clear_plots(self):
        """Clear all plot data when the clear button is pressed"""
        if hasattr(self.parent, 'main_tab') and self.parent.main_tab:
            self.parent.main_tab.clear_plot_data()
            self.parent.main_tab.clear_overlays()

            print("Plot data cleared")
        else:
            print("Main tab not yet initialized")
    
    def exit(self):
        print("Exiting application")
        self.parent.close()

class MainTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent  # Reference to MainWindow for accessing shared data/methods        
        
        # Load UI parameters from config
        ui_config = config.get_ui_params()
        self.scaler = ui_config['scaler']
        self.camera_width = ui_config['max_image_width']
        self.camera_height = ui_config['max_image_height']
        self.label_dim = config.get_label_dimensions()
        
        # Initialize variables needed for video click handling
        self.boundaryFinal = []
        self.clicks = 0
        self.imgCounter = 0
        
        # Initialize tracking arrays
        self.trackX = []# np.zeros(100)  # Store last 100 points
        self.trackY = []# np.zeros(100)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout = self.init_buttons(layout)
        layout = self.init_plots(layout)

        self.setLayout(layout)

    def updateR1(self):
        manual_resistance = self.sliderR1.value() / 100.0 
        if self.parent.ni_process.isRunning() :
            self.parent.send_ni.emit(Task(priority=999, id="1002", data=manual_resistance))
        else:
            self.parent.driver.res_1 = manual_resistance

        self.sliderR1Label.setText(f'Current Value: {manual_resistance}')

    def updateI1(self):
        manual_current = self.sliderI1.value() / 100.0 
        if self.parent.ni_process.isRunning() & self.parent.config_tab.manual_box.isChecked():
            self.parent.send_ni.emit(Task(priority=999, id="1001", data=manual_current))

        self.sliderI1Label.setText(f'Current Value: {manual_current}')

    def init_buttons(self, ll):

        h_layout = QGridLayout()
        h_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        #Resistance slider 
        self.sliderR1 = QSlider(Qt.Orientation.Horizontal, self)
        self.sliderR1.setRange(0,100)
        self.sliderR1.setValue(int(self.parent.args.FirstResis*100))
        self.sliderR1.setSingleStep(1)
        self.sliderR1.setPageStep(1)
        self.sliderR1.setTickPosition(QSlider.TickPosition.TicksRight)
        self.sliderR1.setTickInterval(10)
        self.sliderR1.valueChanged.connect(self.updateR1)

        self.sliderR1Label = QLabel(f'Res 1 Value: {self.sliderR1.value()/100}', self)

        h_layout.addWidget(self.sliderR1Label, 0,0)
        h_layout.addWidget(self.sliderR1, 1, 0)

        #Current slider for tuning
        self.sliderI1 = QSlider(Qt.Orientation.Horizontal, self)
        self.sliderI1.setRange(0,300)
        self.sliderI1.setValue(0)
        self.sliderI1.setSingleStep(1)
        self.sliderI1.setPageStep(10)
        self.sliderI1.setTickPosition(QSlider.TickPosition.TicksRight)
        self.sliderI1.setTickInterval(50)

        self.sliderI1.valueChanged.connect(self.updateI1)
        self.sliderI1Label = QLabel(f'Current Value: {self.sliderI1.value()/100}',self)

        h_layout.addWidget(self.sliderI1Label, 0, 1)
        h_layout.addWidget(self.sliderI1, 1, 1)

        #Display values
        self.CurrentMeasuredLabel = QLabel("Measured Current: {:.2f} A".format(0.00))
        self.CurrentMeasuredLabel.setFixedSize(150,25)
        self.CurrentTargetLabel = QLabel("Target Current: {:.2f} A".format(0.00))
        self.CurrentTargetLabel.setFixedSize(150,25)
        self.BMeasuredLabel = QLabel("Measured B: {:.2f} []".format(0.00))
        self.BMeasuredLabel.setFixedSize(150,25)

        h_layout.addWidget(self.CurrentMeasuredLabel, 0,3)
        h_layout.addWidget(self.CurrentTargetLabel, 1,3)
        h_layout.addWidget(self.BMeasuredLabel, 2,3)

        ll.addLayout(h_layout)

        return ll
    
    def init_plots(self, ll):

        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.plots_tab = {}

        self.plots_tab["currents"] = PlotWidget(plot_dict["currents"]["ylabel"], plot_dict["currents"]["yunits"], plot_dict["currents"]["xlabel"], plot_dict["currents"]["xunits"], plot_dict["currents"]["title"], plot_dict["currents"]["y_limits"])
        for count, i_ in enumerate(plot_dict["currents"]["curves"]):
            self.plots_tab["currents"].add_curve(i_, count)
        # Enable auto-scaling on x-axis - initial range will be set automatically

        self.plots_tab["magnetic_field"] = PlotWidget(plot_dict["magnetic_field"]["ylabel"], plot_dict["magnetic_field"]["yunits"], plot_dict["magnetic_field"]["xlabel"], plot_dict["magnetic_field"]["xunits"], plot_dict["magnetic_field"]["title"], plot_dict["magnetic_field"]["y_limits"])
        for count, i_ in enumerate(plot_dict["magnetic_field"]["curves"]):
            self.plots_tab["magnetic_field"].add_curve(i_, count)
        # Enable auto-scaling on x-axis - initial range will be set automatically

        self.plots_tab["track"] = PlotWidget(plot_dict["track"]["ylabel"], plot_dict["track"]["yunits"], plot_dict["track"]["xlabel"], plot_dict["track"]["xunits"], plot_dict["track"]["title"], plot_dict["track"]["y_limits"])
        for count, i_ in enumerate(plot_dict["track"]["curves"]):
            self.plots_tab["track"].add_curve(i_, count)
        # Set specific range for tracking plot as it uses pixel coordinates
        self.plots_tab["track"].set_range("x", [0, self.camera_width])
        self.plots_tab["track"].set_range("y", [0, self.camera_height])
        # self.plots_tab["track"].set_autoscale()("y")
        
        # Initialize tracking line for trajectory plotting
        self.TrackLine = self.plots_tab["track"].ax.plot(self.trackX, self.trackY, pen=pg.mkPen(color='red', width=2), name='trajectory')

        self.video_widget = VideoWidget(self, self.label_dim[0], self.label_dim[1])
        self.video_widget.setFixedSize(self.label_dim[0], self.label_dim[1])
        self.set_white_screen()

        grid_layout.addWidget(self.plots_tab["currents"].ax, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.plots_tab["magnetic_field"].ax, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.video_widget, 1, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.plots_tab["track"].ax, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        ll.addLayout(grid_layout)

        return ll
    
    def reset_plot_data(self):
        """Reset all plot data across all plot tabs"""
        for i in self.plots_tab.keys():
            for curve in self.plots_tab[i].curve.keys():
                self.plots_tab[i].x_data[curve] = []
                self.plots_tab[i].y_data[curve] = []
                self.plots_tab[i].curve[curve].setData(
                    x=self.plots_tab[i].x_data[curve],
                    y=self.plots_tab[i].y_data[curve]
                )
        
        # Also clear tracking data arrays
        self.trackX = [] #np.zeros(100)
        self.trackY = [] #np.zeros(100)
        if hasattr(self, 'TrackLine'):
            self.TrackLine.setData(self.trackX, self.trackY)

    def clear_plot_data(self, plot_name=None, curve_label=None):

        if plot_name is None:
            # Clear all plots
            self.reset_plot_data()
        else:
            # Clear specific plot
            if plot_name in self.plots_tab:
                self.plots_tab[plot_name].clear_data(curve_label)
                
                # If clearing track plot, also clear tracking arrays
                if plot_name == 'track':
                    self.trackX = [] #np.zeros(100)
                    self.trackY = [] #np.zeros(100)
                    if hasattr(self, 'TrackLine'):
                        self.TrackLine.setData(self.trackX, self.trackY)
            else:
                print(f"Warning: Plot '{plot_name}' not found. Available plots: {list(self.plots_tab.keys())}")
        
        # Clear video overlays
        if hasattr(self, 'video_widget'):
            self.video_widget.clear_overlays()
        self.set_white_screen()


    @pyqtSlot(np.ndarray)
    def setImage(self, image):
        image = image.astype(np.uint8)

        # Update video widget with the image
        if hasattr(self, 'video_widget'):
            self.video_widget.update_frame(image)
            

    def set_white_screen(self):
        """Set initial white screen"""
        background = 1*np.ones((1536, 2048), dtype=np.uint8)
        if hasattr(self, 'video_widget'):
            self.video_widget.update_frame(background)
        else:
            # Fallback for initialization
            h, w = background.shape
            bytesPerLine = 1 * w
            convertToQtFormat = QImage(background, w, h, bytesPerLine, QImage.Format.Format_Grayscale8)
            # p = convertToQtFormat.scaled(self.label_dim[0], self.label_dim[1])
            if hasattr(self, 'label'):
                self.label.setPixmap(QPixmap.fromImage(convertToQtFormat))

    def reset_frames(self):
        """Reset video display and clear overlays"""
        self.set_white_screen()
        
        # Clear video overlays
        if hasattr(self, 'video_widget'):
            self.video_widget.clear_overlays()
        
        # Reset camera tracker
        if hasattr(self, 'cam'):
            self.cam.trackerTool = None
            self.cam.finalboundaries = None
        
        self.boundaryFinal = []

    @pyqtSlot(object)
    def receiveTrackData(self, data):

        # Handle new tracking data format from tracker
        if isinstance(data, dict) and 'position' in data:

            if data.get('lost', False):
                print("Tracking lost")
                return
                
            # Extract position and ROI from new format
            position = data.get('position', None)  # (x, y)
            roi = data.get('roi', None)  # (x1, y1, x2, y2)
                
            display_x = position[0] 
            display_y = position[1] #(self.camera_height - position[1])
            

            if len(self.trackX) >= 50:
                self.trackX = self.trackX[:-1] #np.roll(self.trackX, -1)
                self.trackY = self.trackY[:-1] #np.roll(self.trackY, -1)

            self.trackX.append(display_x)
            self.trackY.append(self.camera_height - display_y)

            # Update plot
            self.TrackLine.setData(self.trackX, self.trackY)
            # Update model coordinates
            if (self.parent.model_process.isRunning()) & (len(self.plots_tab["currents"].y_data["I1_target"])> 0) :
                # try:
                cur_val = float(self.plots_tab["currents"].y_data["I1_target"][-1]) 
                if cur_val > 0:
                    self.parent.send_model.emit(Task(priority=999, id="1001", data=[display_x, self.camera_height - display_y, cur_val]))
                elif cur_val == 0:
                    self.parent.send_model.emit(Task(priority=999, id="1002", data=[-1, -1, -1]))
                # except:
                #     print("Failed")
            # Draw tracking rectangle if ROI is available
            if roi and hasattr(self, 'video_widget'):
                x1, y1, x2, y2 = roi

                # Scale ROI to display coordinates with Y coordinate inversion
                display_x1 = x1#x1 
                display_y1 = y1#(camera_height - y1) / scaler
                
                display_x2 = x2#x2 / scaler
                display_y2 = y2#(camera_height - y2) / scaler
                
                
                if display_y1 > display_y2:
                    display_y1, display_y2 = display_y2, display_y1
                
                
                self.clear_tracking_rectangle()
                
                
                self.video_widget.add_rectangle(display_x1, display_y1, display_x2, display_y2, 
                                              color=Qt.GlobalColor.green, width=2)
            
                
                self.video_widget.add_trajectory_point(display_x, display_y, 
                                                        color=Qt.GlobalColor.blue, radius=2)
                
                
                if len(self.video_widget.overlay_items) > 150:  # Keep last 150 points
                    

                    # Remove oldest trajectory points (keep rectangles)
                    items_to_remove = []
                    for item in self.video_widget.overlay_items[:30]:  # Remove oldest 30
                        if hasattr(item, 'rect'):  # It's an ellipse (trajectory point)
                            items_to_remove.append(item)
                    
                    for item in items_to_remove:
                        self.video_widget.scene.removeItem(item)
                        self.video_widget.overlay_items.remove(item)
        
        else:
            print("tracker send incorrect data package")


    def clear_tracking_rectangle(self):
        """Remove tracking rectangles"""
        if hasattr(self, 'video_widget'):
            items_to_remove = []
            for item in self.video_widget.overlay_items:
                
                if hasattr(item, 'rect'):
                    items_to_remove.append(item)
            
            for item in items_to_remove:
                self.video_widget.scene.removeItem(item)
                self.video_widget.overlay_items.remove(item)

    def handle_video_click(self, x, y):
        """Handle clicks on video widget for tracker rectangle selection"""
        
        if self.clicks == 0:
            self.boundaryFinal = []

        self.boundaryFinal.append([(x, y)])
        self.clicks += 1
        
        if self.clicks == 2:
            self.x1 = self.boundaryFinal[0][0][0] 
            self.x2 = self.boundaryFinal[1][0][0]
            self.y1 = self.boundaryFinal[0][0][1]
            self.y2 = self.boundaryFinal[1][0][1]


            self.video_widget.clear_overlays()
            self.video_widget.add_rectangle(self.x1, self.y1, self.x2, self.y2)
            self.clicks = 0

    def update_str_field(self, msg):
        """Update status message field"""
        self.parent.top_row.printfield.setText(msg)

class AnalysisTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent  # Reference to MainWindow for accessing shared data/methods

        layout = QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(layout)


class ConfigTab(QWidget):
    def __init__(self, parent):
        super().__init__()

        self.parent = parent  # Reference to MainWindow for accessing shared data/methods
        self.param = {}

        multi_feedback_layout = QVBoxLayout()
        multi_feedback_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Create feedback mode combo box with fixed width
        feedback_label = QLabel("Feedback Mode")
        feedback_label.setFixedWidth(120)
        self.combobox_feedback = QComboBox(self)
        self.combobox_feedback.setFixedWidth(120)
        self.combobox_feedback.setFixedHeight(25)

        for item in multi_choices["feedback_mode"]:
            self.combobox_feedback.addItem(item)

        multi_feedback_layout.addWidget(feedback_label)
        multi_feedback_layout.addWidget(self.combobox_feedback)

        multi_test_layout = QVBoxLayout()
        multi_test_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Create test mode combo box w
        test_label = QLabel("Test Mode")
        test_label.setFixedWidth(120)
        self.combobox_test = QComboBox(self)
        self.combobox_test.setFixedWidth(120)
        self.combobox_test.setFixedHeight(25)

        for item in multi_choices["test_mode"]:
            self.combobox_test.addItem(item)

        multi_test_layout.addWidget(test_label)
        multi_test_layout.addWidget(self.combobox_test)

        multi_waveform_layout = QVBoxLayout()
        multi_waveform_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Create waveform combo box 
        waveform_label = QLabel("Waveform")
        waveform_label.setFixedWidth(120)
        self.combobox_wave = QComboBox(self)
        self.combobox_wave.setFixedWidth(120)
        self.combobox_wave.setFixedHeight(25)

        for item in multi_choices["waveform"]:
            self.combobox_wave.addItem(item)

        multi_waveform_layout.addWidget(waveform_label)
        multi_waveform_layout.addWidget(self.combobox_wave)

        self.saving_box = QCheckBox("Save")
        self.saving_box.setCheckState(Qt.CheckState.Unchecked)
        self.saving_box.setFixedWidth(120)

        self.manual_box = QCheckBox("Manual Input")
        self.manual_box.setCheckState(Qt.CheckState.Unchecked)
        self.manual_box.setFixedWidth(120)

        self.tracker_box = QCheckBox("Tracker")
        self.tracker_box.setCheckState(Qt.CheckState.Unchecked)
        self.tracker_box.setFixedWidth(120)

        self.emp_box = QCheckBox("Empirical Model")
        self.emp_box.setCheckState(Qt.CheckState.Unchecked)
        self.emp_box.setFixedWidth(120)

        layout = QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addLayout(multi_feedback_layout, 0, 0)
        layout.addLayout(multi_test_layout, 0, 1)
        layout.addLayout(multi_waveform_layout, 0, 2)

        layout.addWidget(self.saving_box, 1, 0)
        layout.addWidget(self.manual_box, 2, 0)
        layout.addWidget(self.tracker_box, 3, 0)
        layout.addWidget(self.emp_box, 4, 0)

        # Load default values from configuration
        ui_defaults = config.get_ui_defaults()
        
        layout.addLayout(self.create_box_pair("offset", ui_defaults["offset"]), 1, 1)
        layout.addLayout(self.create_box_pair("grad", ui_defaults["grad"]), 1, 2)
        layout.addLayout(self.create_box_pair("freq", ui_defaults["freq"]), 1, 3)
        layout.addLayout(self.create_box_pair("start_time", ui_defaults["start_time"]), 2, 1)
        layout.addLayout(self.create_box_pair("end_time", ui_defaults["end_time"]), 2, 2)
        layout.addLayout(self.create_box_pair("total_time", ui_defaults["total_time"]), 2, 3)
        layout.addLayout(self.create_box_pair("peak_distance", ui_defaults["peak_distance"]), 3,1)
        layout.addLayout(self.create_box_pair("n_peaks", ui_defaults["n_peaks"]), 3, 2)
        layout.addLayout(self.create_box_pair("FrameRate (fps)", ui_defaults["FrameRate (fps)"]), 3, 3)
        layout.addLayout(self.create_box_pair("exposureTime (ms)", ui_defaults["exposureTime (ms)"]), 4, 1)
        layout.addLayout(self.create_box_pair("gain", ui_defaults["gain"]), 4, 2)
        layout.addLayout(self.create_box_pair("pixel_size", ui_defaults["pixel_size"]), 4, 3)
        layout.addLayout(self.create_box_pair("objective", ui_defaults["objective"]), 5, 1)
        layout.addLayout(self.create_box_pair("light_path_zoom", ui_defaults["light_path_zoom"]), 5, 2)
        layout.addLayout(self.create_box_pair("camera_adapter_zoom", ui_defaults["camera_adapter_zoom"]), 5, 3)
        layout.addLayout(self.create_box_pair("camera_height", ui_defaults["camera_height"]), 6, 1)
        layout.addLayout(self.create_box_pair("camera_width", ui_defaults["camera_width"]), 6, 2)

        self.setLayout(layout)

    def create_box_pair(self, name, value):
        _layout = QVBoxLayout()
        

        label = QLabel(name)
        label.setFixedWidth(120)
        label.setWordWrap(True) 
        _layout.addWidget(label)

        self.param[name] = QLineEdit()
        self.param[name].setText(value)
        self.param[name].setFixedWidth(120)
        self.param[name].setFixedHeight(25)
        _layout.addWidget(self.param[name])

        return _layout


class PlotWidget(QWidget):

    def __init__(self,  ylabel, yunits,  xlabel, xunits, title, ylimits):

        super().__init__()
        
        # Set fixed size for consistent layout
        self.setFixedWidth(400)
        self.setFixedHeight(300)

        self.ax = pg.PlotWidget()

        self.view_box = self.ax.getViewBox()
        self.view_box.setMouseEnabled(x=True, y=True)  # Allow user zoom/pan
        
        # Configure auto-ranging for optimal display
        self.view_box.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        self.view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)  # Keep Y fixed to limits
        
        # Set automatic scaling behavior
        self.view_box.setAutoVisible(x=True, y=False)
        self.ax.setAutoVisible(x=True, y=False)

        self.ax.setTitle(title)
        self.ax.addLegend(offset=(10, 10))
        self.set_range("y", ylimits)


        font = QFont()
        font.setPointSize(4) 
        self.legend = self.ax.addLegend()

        for item in self.legend.items:
            label = item[1] 
            label.setFont(font)

        self.y_data = {}
        self.x_data = {}
        self.curve = {}

        self.ax.setLabel("left", ylabel, yunits)
        self.ax.setLabel("bottom",  xlabel, xunits)

    def set_autoscale(self):
        # Configure auto-ranging for optimal display
        self.view_box.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        self.view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)  
        
        # Set automatic scaling behavior
        self.view_box.setAutoVisible(x=True, y=True)
        self.ax.setAutoVisible(x=True, y=True)

    def set_range(self, label, value):
        if label == "x":
            self.view_box.setRange(xRange=(value[0], value[1]), padding=0)
        elif label == "y":
            self.view_box.setRange(yRange=(value[0], value[1]), padding=0)

    def update_axis(self,x, x_unit, y, y_unit):

        self.ax.setLabel("left", y, y_unit)
        self.ax.setLabel("bottom",  x, x_unit)

    def update_legend(self, legend, num):

        curve_names = list(self.curve.keys())
        if legend=="":
            self.legend.removeItem(curve_names[num])
        else:
            self.legend.removeItem(curve_names[num])
            self.legend.addItem(self.curve[curve_names[num]], legend)

    def add_curve(self, label, number):

        self.x_data[label] = []
        self.y_data[label] = []
        self.curve[label] = self.ax.plot(x = self.x_data[label], y=self.y_data[label], pen = pg.mkPen(color=colors_list[number]), name =label)


    def update_plot(self, x_value, y_value, label):

        max_points = 1000 
        if len(self.x_data[label]) >= max_points:
            self.x_data[label] = self.x_data[label][1:]
            self.y_data[label] = self.y_data[label][1:]

        self.y_data[label].append(y_value)
        self.x_data[label].append(x_value)
        self.curve[label].setData(x=self.x_data[label], y=self.y_data[label])
        
        if len(self.x_data[label]) > 1:
            x_min = min(self.x_data[label])
            x_max = max(self.x_data[label])
            x_range = x_max - x_min if x_max > x_min else 1
            margin = x_range * 0.05  # 5% margin
            self.ax.setXRange(x_min - margin, x_max + margin, padding=0)

    def clear_data(self, label=None):
        """
        Clear plot data for specific curve(s) or all curves
        """
        if label is None:
            # Clear all curves
            for curve_label in self.curve.keys():
                self.x_data[curve_label] = []
                self.y_data[curve_label] = []
                self.curve[curve_label].setData(x=[], y=[])
            # Reset auto-range after clearing all data
            self.ax.enableAutoRange()
        else:
            # Clear specific curve
            if label in self.curve:
                self.x_data[label] = []
                self.y_data[label] = []
                self.curve[label].setData(x=[], y=[])
                # Enable auto-range if this was the only curve with data
                has_data = any(len(self.x_data[lbl]) > 0 for lbl in self.curve.keys())
                if not has_data:
                    self.ax.enableAutoRange()
            else:
                print(f"Warning: Curve '{label}' not found in plot")



class VideoWidget(QGraphicsView):
    
    def __init__(self, parent, label_width , label_height):
        super().__init__(parent)
        
        # Create scene and set it
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.camera_width = int(parent.camera_width)
        self.camera_height = int(parent.camera_height)

        self.label_width = int(label_width)
        self.label_height = int(label_height)

        self.w_map = self.camera_width / self.label_width
        self.h_map = self.camera_height / self.label_height

        # Video display item
        self.video_item = QGraphicsPixmapItem()
        self.scene.addItem(self.video_item)
        
        self.overlay_items = []
        
        # Configure view for performance
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Store original image size for coordinate mapping
        self.original_size = None
        
    def mousePressEvent(self, event):
        "Handle mouse clicks for tracking rectangle selection"
        
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())

            pixmap_rect = self.video_item.boundingRect()

            if pixmap_rect.contains(scene_pos):
                # Calculate relative position within the pixmap
                rel_x = (scene_pos.x() - pixmap_rect.x()) / pixmap_rect.width()
                rel_y = (scene_pos.y() - pixmap_rect.y()) / pixmap_rect.height()
                
                # Map to original image coordinates
                orig_x = rel_x * self.camera_width  # width
                orig_y = rel_y * self.camera_height  # height
                
                # Emit signal to parent with original coordinates
                if hasattr(self.parent(), 'handle_video_click'):
                    self.parent().handle_video_click(orig_x, orig_y)
            
        super().mousePressEvent(event)
    
    def update_frame(self, image, clear_overlays=False):
        # Store original image size
        if clear_overlays:
            self.clear_overlays()
            
        if len(image.shape) == 2:  
            h, w = image.shape
            bytes_per_line = w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:  # Color (RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Convert to pixmap and update
        pixmap = QPixmap.fromImage(q_image)    
        scaled_pixmap = pixmap.scaled(self.label_width, self.label_height, Qt.AspectRatioMode.KeepAspectRatio,)
        self.video_item.setPixmap(scaled_pixmap)
        
    
    def clear_overlays(self):
        """Remove all overlay items"""
        for item in self.overlay_items:
            self.scene.removeItem(item)

        self.overlay_items.clear()
    
        self.scene.update()
        self.update()
    
    def add_rectangle(self, x1, y1, x2, y2, color=Qt.GlobalColor.red, width=2):
        """Add rectangle overlay in original image coordinates"""

        pixmap_rect = self.video_item.boundingRect()
        
        scene_x1 = pixmap_rect.x() + (x1 / self.camera_width) * pixmap_rect.width()
        scene_y1 = pixmap_rect.y() + (y1 / self.camera_height) * pixmap_rect.height()

        scene_x2 = pixmap_rect.x() + (x2 / self.camera_width) * pixmap_rect.width()
        scene_y2 = pixmap_rect.y() + (y2 / self.camera_height) * pixmap_rect.height()

        # Create rectangle item
        rect_item = QGraphicsRectItem(scene_x1, scene_y1, scene_x2 - scene_x1, scene_y2 - scene_y1)
        pen = QPen(color, width)
        rect_item.setPen(pen)
        rect_item.setBrush(QBrush())  
        
        self.scene.addItem(rect_item)
        self.overlay_items.append(rect_item)
    
        self.scene.update()
        self.update()
        
        return rect_item
    
    def add_trajectory_point(self, x, y, color=Qt.GlobalColor.green, radius=3):
        """Add trajectory point in original image coordinates"""
        if not self.original_size or not self.video_item.pixmap():
            return
            
        # Map to scene coordinates
        pixmap_rect = self.video_item.boundingRect()
        scene_x = pixmap_rect.x() + (x / self.camera_width) * pixmap_rect.width()
        scene_y = pixmap_rect.y() + (y / self.camera_height) * pixmap_rect.height()
        
        # Create circle item
        circle_item = self.scene.addEllipse(scene_x - radius, scene_y - radius, 
                                           radius * 2, radius * 2,
                                           QPen(color), QBrush(color))
        self.overlay_items.append(circle_item)
        print("adding trajectory point", x, y)
        
        # Force immediate update of the graphics view
        self.scene.update()
        self.update()
        
        return circle_item
    
import numpy as np
import time

class EMAFilter:
    """
    Exponential Moving Average Filter
    
    """
    
    def __init__(self, alpha=0.1, initial_value=0.0):
        """
        Initialize EMA filter
        """
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
            
        self.alpha = alpha
        self.filtered_value = initial_value
        self.is_initialized = False
        
    def update(self, new_value):

        if not self.is_initialized:
            # First measurement - initialize filter
            self.filtered_value = new_value
            self.is_initialized = True
        else:
            # Apply EMA formula: y[n] = α * x[n] + (1-α) * y[n-1]
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
            
        return self.filtered_value
    
    def get_value(self):
        return self.filtered_value
    
    def reset(self, initial_value=0.0):
        self.filtered_value = initial_value
        self.is_initialized = False
    
    def set_alpha(self, alpha):
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
    
    @classmethod
    def from_time_constant(cls, time_constant, sample_time, initial_value=0.0):
        """
        Create EMA filter from time constant
        """
        alpha = 1 - np.exp(-sample_time / time_constant)
        return cls(alpha, initial_value)


class PIController:
    
    def __init__(self, kp=1.0, ki=0.1, setpoint=0.0, output_limits=None, sample_time=0.01):
        """
        Initialize PI controller
        """
        self.kp = kp
        self.ki = ki
        self.setpoint = setpoint
        self.sample_time = sample_time
        
        # Internal state
        self.integral_term = 0.0
        self.last_time = None
        self.last_error = 0.0
        
        # Output limits
        self.output_min, self.output_max = output_limits
            
        # For monitoring
        self.last_output = 0.0
        self.proportional_term = 0.0
        
    def update(self, measurement, setpoint, dt=None):
        """Standard PI with simple anti-windup and proper dt handling."""
        self.setpoint = setpoint
        current_time = time.time()
        dt = dt if dt is not None else self.sample_time

        # Error and proportional term
        error = self.setpoint - measurement
        self.proportional_term = self.kp * error

        # 2.75 -> amplifier amplification arbitory
        unsat_output = 2.75*(setpoint) + self.proportional_term + self.ki * (self.integral_term + error * dt)

        #  saturation
        would_saturate_high = (self.output_max is not None) and (unsat_output > self.output_max) and (error > 0)
        would_saturate_low = (self.output_min is not None) and (unsat_output < self.output_min) and (error < 0)
        if not (would_saturate_high or would_saturate_low):
            self.integral_term += error * dt

        # Clamp
        output = unsat_output
        if self.output_max is not None:
            output = min(output, self.output_max)
        if self.output_min is not None:
            output = max(output, self.output_min)

        # Store for next iteration
        self.last_time = current_time
        self.last_error = error
        self.last_output = output

        return output
    
    def set_gains(self, kp=None, ki=None):
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
    
    def set_output_limits(self, limits):
        self.output_limits = limits
        if limits:
            self.output_min, self.output_max = limits
        else:
            self.output_min, self.output_max = None, None
    
    def reset(self):

        self.integral_term = 0.0
        self.last_time = None
        self.last_error = 0.0
        self.last_output = 0.0
        self.proportional_term = 0.0
    
    
    @classmethod
    def from_ziegler_nichols(cls, ku, tu, control_type='PI', **kwargs):
        """
        Autotuning, implemented but not used, For fun""
        """
        if control_type == 'PI':
            kp = 0.45 * ku
            ki = 0.54 * ku / tu
        elif control_type == 'P':
            kp = 0.5 * ku
            ki = 0.0
        else:
            raise ValueError("control_type must be 'PI' or 'P'")
            
        return cls(kp=kp, ki=ki, **kwargs)
