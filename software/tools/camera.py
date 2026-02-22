
from pypylon import pylon
from pypylon import genicam

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, pyqtSlot, QCoreApplication
from PyQt6.QtGui import QImage
import numpy as np
import time

import subprocess
from datetime import datetime

from queue import Queue
from tools.tracker import TrackerThread
from tools.tools import Task
from tools.config import get_config

config = get_config()



class BaslerCam(QObject):

    send_image = pyqtSignal(np.ndarray)
    frame_for_tracker = pyqtSignal(Task)

    send_data = pyqtSignal(object)

    send_position_data = pyqtSignal(object)
    basler_print_str = pyqtSignal(str) #self.basler_print_str.emit(pos)

    def __init__(self, args, parent=None):
        super().__init__(parent)
        
        
        self.args = args
        self.parent = parent
        
        self.label_dim = config.get_label_dimensions()

        self.trackFlag = False
        self.saveFlag = False
        self.mode = "0"
        self.start_time = None
        self._running = False  # Initialize running flag
        self.connected = False
        self.is_emulated = False  # Track whether this is an emulated camera
        
        try:
            # First, try to enumerate all available devices (real and emulated)
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            print(f"Found {len(devices)} camera device(s)")
            for i, device in enumerate(devices):
                device_info = device.GetFriendlyName()
                model = device.GetModelName()
                print(f"  Device {i}: {device_info} ({model})")
            
            if not devices:
                print("No cameras found (real or emulated)")
                self.connected = False
                return None
            
            # Use first available device (real or emulated)
            self.cam = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            self.converter = pylon.ImageFormatConverter()
            
            # Check if this is an emulated camera
            device_name = devices[0].GetFriendlyName()
            if "Emulation" in device_name:
                print(f"Connected to emulated camera: {device_name}")
                self.is_emulated = True
            else:
                print(f"Connected to real camera: {device_name}")
                self.is_emulated = False
                
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.connected = False
            return None
        
        # Configure converter for Mono8 format (since we'll set camera to Mono8)
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        try:
            self.cam.Open()
        except Exception as e:
            print(f"Error opening Basler camera: {e}")
            self.connected = False
            return None
        try:
            self.cfg_Camera()
        except Exception as e:
            print(f"Error configuring Basler camera: {e}")
            self.connected = False
            return None
        
        # Initialize video writer (will be started when needed)
        self.video_writer = None
        self.recording_video = False
        
        self.connected = True


    def _reset_variables(self):
        self._running = False
        self.trackFlag = False
        self.saveFlag = False
        self.mode = "0"
        self.start_time = None
        
        # Stop video recording if active
        if hasattr(self, 'recording_video') and self.recording_video:
            self.stop_video_recording()
        
        # Clear existing queue if it exists
        if hasattr(self, 'track_que'):
            while not self.track_que.empty():
                try:
                    self.track_que.get_nowait()
                except:
                    break
        
        
        # Reset tracker if it exists
        if hasattr(self, 'tracker'):
            self.tracker._reset_variables()
        
        # Process any pending Qt events to clear signal/slot queues
        QCoreApplication.processEvents()

    def cfg_Camera(self):
        """Configure Basler camera parameters with proper error handling"""
        
        try:
            # Set exposure time (convert from milliseconds to microseconds)
            exposure_us = self.args.exposure * 1000  # ms to μs
            self.cam.ExposureTime.SetValue(exposure_us)
        except Exception as e:
            print(f"Could not set exposure time: {e}")
        
        try:
            # Enable frame rate control and set it
            self.cam.AcquisitionFrameRateEnable.SetValue(True)
            self.cam.AcquisitionFrameRate.SetValue(self.args.framerate)
        except Exception as e:
            print(f"Could not set frame rate: {e}")
        
        try:
            # Set pixel format to Mono8 (grayscale)
            self.cam.PixelFormat.SetValue("Mono8")
        except Exception as e:
            print(f"Could not set pixel format: {e}")
        
        try:
            self.cam.MaxNumBuffer.SetValue(1000)
        except Exception as e:
            print(f"Could not set buffer count: {e}")
        
        # Get current settings for verification
        try:
            self.exposureTime = self.cam.ExposureTime.GetValue()*1e-3
            self.width = self.cam.Width.GetValue()
            self.height = self.cam.Height.GetValue()
            self.gain = self.cam.Gain.GetValue()
            self.pixelFormat = self.cam.PixelFormat.GetValue()
            self.FrameRate = self.cam.AcquisitionFrameRate.GetValue()
        except Exception as e:
            print(f"Could not read camera parameters: {e}")
            # Set defaults for emulated camera
            self.exposureTime = self.args.exposure
            self.width = 640  # Common emulated camera size
            self.height = 480
            self.gain = 1.0
            self.pixelFormat = "Mono8"
            self.FrameRate = self.args.framerate
        
        # Calculate frame count for timed acquisition
        self.frameCount = 1000  
        
        camera_type = "Emulated" if self.is_emulated else "Real"
        print(f"{camera_type} camera configured: {self.width}x{self.height}, {self.FrameRate} fps, {self.exposureTime} ms exposure")
        
    def start_video_recording(self):
        """Start video recording if saveFlag is True"""
        if not self.saveFlag or self.recording_video:
            return False
            
        # Generate filename with timestamp
        filename = f"./results/basler.mp4"
        print("debug", self.width, self.height)
        # Create video writer
        self.video_writer = QtFastVideoWriter(
            filename, 
            self.width, 
            self.height, 
            self.FrameRate
        )
        
        # Start recording
        if self.video_writer.start_recording():
            self.recording_video = True
            print(f"Started video recording: {filename}")
            return True
        else:
            print("Failed to start video recording")
            self.video_writer = None
            return False
    
    def stop_video_recording(self):
        """Stop video recording"""
        if self.recording_video and self.video_writer:
            self.video_writer.stop_recording()
            self.recording_video = False
            self.video_writer = None
            print("Video recording stopped")
    
    def record_frame_to_video(self, frame):
        """Record a single frame to video if recording is active"""
        if self.recording_video and self.video_writer:
            self.video_writer.write_frame(frame)
    
    def _update_params(self, param):

        self.trackFlag = param["trackFlag"]
        self.start_time = param["timestamp_sync"]
        self.saveFlag = param["Flag_save"]

        # Update frame rate if different
        if param["FrameRate (fps)"] != self.FrameRate:
            try:
                self.cam.AcquisitionFrameRate.SetValue(param["FrameRate (fps)"])
                self.FrameRate = param["FrameRate (fps)"]
            except Exception as e:
                print(f"Could not update frame rate: {e}")
                self.FrameRate = param["FrameRate (fps)"]  # Update locally even if camera fails
                
        # Update exposure time if different        
        if param["exposureTime (ms)"] != self.exposureTime:
            try:
                exposure_us = param["exposureTime (ms)"] * 1000  # ms to μs
                self.cam.ExposureTime.SetValue(exposure_us)
                self.exposureTime = param["exposureTime (ms)"]
            except Exception as e:
                print(f"Could not update exposure time: {e}")
                self.exposureTime = param["exposureTime (ms)"]  # Update locally even if camera fails
                
        # Update gain if different
        if param["gain"] != self.gain:
            try:
                self.cam.Gain.SetValue(param["gain"])
                self.gain = param["gain"]
            except Exception as e:
                print(f"Could not update gain: {e}")
                self.gain = param["gain"]  # Update locally even if camera fails

        self.frameCount = int(param["total_time"] * self.FrameRate)

        print("updated frame count:", self.frameCount)
        # self.cam.MaxNumBuffer.SetValue(self.frameCount)

    def _init_tracker(self):
        # try:
        print("Setting starting boundaries for tracker...", self.finalboundaries)
        self.tracker = TrackerThread(self.saveFlag)
        # Initialize tracker with boundaries
        boundary_param = type('obj', (object,), {
            'w': self.width,
            'h': self.height,
            'k': 1.0,
            'x_scale': self.width/self.label_dim[0],
            'y_scale': self.height/self.label_dim[1],
            'x': self.finalboundaries[0][0][0],
            'x2': self.finalboundaries[1][0][0],
            'y': self.finalboundaries[0][0][1],
            'y2': self.finalboundaries[1][0][1]
        })

        self.tracker._init_params(boundary_param)
        
        # Connect tracker signals properly
        self.tracker.tracking_data.connect(self.forward_TrackData, Qt.ConnectionType.QueuedConnection)
        self.frame_for_tracker.connect(self.tracker.receive_frame)

        self.tracker_process = QThread()  # Use imported QThread
        self.tracker.moveToThread(self.tracker_process)
        self.tracker_process.started.connect(self.tracker.run)

    def run(self):
        """
        Modes:
            "0" - idle/snap
            "1" - livestream
            "2" - measurement/recording
        """
        print("Camera worker thread started")
        self._running = True
        
        # Persistent loop
        try:
            if self.mode == "0":
                # Idle mode - just wait
                time.sleep(0.1)
                
            elif self.mode == "1":
                # Livestream mode
                self.livestream()
                # After livestream ends, go back to idle
                self.mode = "0"
                
            elif self.mode == "2":
                # Measurement/recording mode
                
                if self.trackFlag: 
                    try:
                        print("Starting tracker process...")
                        self.tracker_process.start()
                    except Exception as e:
                        print("failed to start tracker")
                        self.basler_print_str.emit(f"Failed to start tracker: {e}")
                        self.trackFlag = False  # Disable tracking if initialization fails

                self.record_measurement()

                if self.trackFlag:
                        self.frame_for_tracker.emit(Task(priority=999, id="1000", data=[]))
                # After recording ends, go back to idle
                self.mode = "0"
                
            else:
                # Unknown mode, wait
                time.sleep(0.1)
        except Exception as e:
            self.basler_print_str.emit(f"Camera error: {e}")
            self.mode = "0"
            time.sleep(0.1)

        QCoreApplication.processEvents()

    def snapImage(self):

        self.basler_print_str.emit("Image snapped")
        with self.cam.GrabOne(1000) as res:
            image = self.converter.Convert(res)
            self.frame = image.GetArray()
            res.Release()
            image.Release()

        #Emit frame
        self.send_image.emit(self.frame)

    def livestream(self):
        """
        Stream video until stop command
        """
        
        self.basler_print_str.emit("Live streaming started")
        frame_count = 0
        
        try:
            # Start grabbing with latest image strategy for real-time streaming
            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            
            while self.cam.IsGrabbing() and self._running:
                QCoreApplication.processEvents()
                try:
                    # Retrieve result with shorter timeout for responsive stopping
                    grab = self.cam.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
                    
                    if grab.GrabSucceeded():
                        # Convert image efficiently
                        image = self.converter.Convert(grab)
                        frame = image.GetArray()
                        
                        # Emit frame to UI
                        self.send_image.emit(frame)
                        frame_count += 1
                        
                        # Release converted image
                        image.Release()
                        
                        # Always release grab buffer
                        grab.Release()
                        
                    else:
                        self.basler_print_str.emit("Grab failed")
                        grab.Release()
                        break
                        
                except Exception as e:
                    self.basler_print_str.emit(f"Error during grab: {e}")
                    if 'grab' in locals():
                        try:
                            grab.Release()
                        except:
                            pass
                    break
            
            # Clean shutdown
            if self.cam.IsGrabbing():
                self.cam.StopGrabbing()
                
            self.basler_print_str.emit(f"Live streaming stopped. Processed {frame_count} frames")
            
        except Exception as e:
            self.basler_print_str.emit(f"Livestream error: {e}")
            if self.cam.IsGrabbing():
                try:
                    self.cam.StopGrabbing()
                except:
                    pass
        
        return 0
    


    def record_measurement(self):
        """
        Stream video until stop command
        """
        
        self.basler_print_str.emit("Measurement recording started")
        frame_count = 0
        timestamps = []
        
        # Start video recording if saveFlag is True
        if self.saveFlag:
            self.start_video_recording()
        
        try:
            # Start grabbing with latest image strategy for real-time streaming
            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            
            while self.cam.IsGrabbing() and self._running and frame_count < self.frameCount:
                QCoreApplication.processEvents()
                try:
                    # Use shorter timeout for more responsive stopping
                    grab = self.cam.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
                    
                    if grab.GrabSucceeded():
                        # Convert image efficiently
                        image = self.converter.Convert(grab)
                        frame = image.GetArray()
                        timestamp = time.time() - self.start_time
                        timestamps.append(timestamp)

                        if self.trackFlag and frame_count % 1 == 0:
                            # print("emitting frame to tracker", timestamp)
                            self.frame_for_tracker.emit(Task(priority=999, id="1001", data=[frame, timestamp]))

                        # Record frame to video if recording is active
                        if self.recording_video:
                            self.record_frame_to_video(frame)

                        # Emit frame to UI
                        self.send_image.emit(frame)
                        frame_count += 1
                        
                        # Release 
                        image.Release()
                        grab.Release()
                        
                        # Check _running flag more frequently 
                        if not self._running:
                            self.basler_print_str.emit("Measurement stopped by user")
                            break
                        
                    else:
                        self.basler_print_str.emit("Grab failed")
                        grab.Release()
                        break
                        
                except Exception as e:

                    if "timeout" in str(e).lower() and not self._running:
                        self.basler_print_str.emit("Measurement stopped by user")
                        break
                    else:
                        self.basler_print_str.emit(f"Error during grab: {e}")

                    if 'grab' in locals():
                        try:
                            grab.Release()
                        except:
                            pass
                    
                    if not self._running:
                        break
            
            # Clean shutdown
            if self.cam.IsGrabbing():
                self.cam.StopGrabbing()

            # Stop video recording
            if self.recording_video:
                self.stop_video_recording()

            if self.saveFlag:
                np.save(f"./results/timestamps.npy", np.array(timestamps))

            if self.trackFlag:
                self.frame_for_tracker.emit(Task(priority=999, id="1000", data=frame))
                self.tracker_process.quit()
                print("Waiting for tracker save process to finish...")
                try:
                    self.tracker_process.wait(3000)
                except Exception:
                    self.ni_process.terminate()

            self.basler_print_str.emit(f"Live streaming stopped. Processed {frame_count} frames")
            
        except Exception as e:
            self.basler_print_str.emit(f"Livestream error: {e}")
            if self.cam.IsGrabbing():
                try:
                    self.cam.StopGrabbing()
                except:
                    pass
        
        return 0

    @pyqtSlot(object)
    def forward_TrackData(self, tracking_data):
        self.send_data.emit(tracking_data)

    @pyqtSlot(object)
    def receive_main(self, data):
        """Handle commands from main thread"""
        if data.id == "1000":
            # Stop current operation
            self._running = False
            self.mode = "0"
        elif data.id == "flush":
            # Flush command to clear all pending messages
            print("Camera received flush command - clearing queue and forwarding to tracker")
            # Clear track queue
            while not self.track_que.empty():
                try:
                    self.track_que.get_nowait()
                except:
                    break
            
            # Forward flush command to tracker if it exists
            if hasattr(self, 'tracker'):
                self.frame_for_tracker.emit(data)
            
            QCoreApplication.processEvents()
        elif data.id == "terminate":
            # Terminate the worker loop completely
            print("receved terminate")
            self._running = False
            self.mode = "0"
            self._terminate = True
        elif data.id == "mode":
            # Set mode directly
            self.mode = str(data.data)


import ffmpeg

class QtFastVideoWriter(QObject):
    """ FFmpeg subprocess"""
    
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, filename, width, height, fps=30, codec='libx264', preset='ultrafast'):
        super().__init__()
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        

        self.running = False
        
    def start_recording(self):
        """Start the video recording process"""
        
        self.process = ( 
            ffmpeg 
            .input('pipe:', format='rawvideo', framerate=self.fps, pix_fmt='rgb24', s='{}x{}'
            .format(self.width, self.height)) 
            .output(self.filename, pix_fmt='yuv420p').overwrite_output() 
            .run_async(pipe_stdin=True) 
            )

        self.running = True
        print(f"Started video recording to {self.filename}")

        return True
    
    def write_frame(self, frame):
        """Write a frame to the video (non-blocking)"""
        if not self.running or not self.process:
            return False
            
        try:
            frame = np.stack((frame, frame, frame), axis=-1)
            # Write frame data directly to FFmpeg stdin
            self.process.stdin.write(frame.tobytes())
            return True
            
        except Exception as e:
            print(f"Error writing frame: {e}")
            return False
    
    def stop_recording(self):
        """Stop video recording"""
        if not self.running:
            return
            
        self.running = False
        
        if self.process:
            print("Waiting recorder to finish...")
            try:
                # Close stdin to signal end of input
                self.process.stdin.close()
                # Wait for FFmpeg to finish with timeout
                self.process.wait(timeout=320)
                print(f"Video recording finished: {self.filename}")
                
            except subprocess.TimeoutExpired:
                print("FFmpeg didn't finish in time, terminating...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    
            except Exception as e:
                self.error.emit(f"Error stopping video recording: {e}")
            
            finally:
                self.process = None
                
        self.finished.emit()