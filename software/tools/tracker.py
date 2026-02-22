import cv2
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, QCoreApplication
import numpy as np
import time
from tools.tools import Task, LatestQueue

class TrackerThread(QThread):
    # Define signals for different types of data
    tracking_data = pyqtSignal(dict)      # Complete tracking info
    

    def __init__(self, save, parent=None):
        super().__init__(parent)
        #rescaled = 640x480 and init 2048x1536
        self.save = save
        self.init_all()

    def init_all(self):
        self.w = None
        self.h = None
        self.k = None

        self.x_scale = None
        self.y_scale = None

        # change from screen coordinates to image coordinates
        self.x = None
        self.x2 = None
        self.y = None
        self.y2 = None

        # Initialize tracking data storage
        self.Dict = {"x": [], "y": [], "t": []}
        self.t = 0
        self.frame_count = 0
        
        # Tracking state
        self.frameCrop = None
        self.binary = None
        self.tracking_quality = 0.0
        self.lost_frames = 0
        self.max_lost_frames = 5
        self.min_area = 10 

        self.que = LatestQueue()
        self._running = False

    def _reset_variables(self):
        """Reset tracker state and clear queues"""
        self._running = False
        self.frame_count = 0
        self.tracking_quality = 0.0
        self.lost_frames = 0
        
        # Clear the processing queue
        while not self.que.empty():
            try:
                self.que.get_nowait()
            except:
                break
        
        # Reset tracking state
        self.frameCrop = None
        self.binary = None

    def _init_params(self, param):
        #rescaled = 640x480 and init 2048x1536
        self.w = param.w
        self.h = param.h
        self.k = param.k

        self.x_scale = param.x_scale #2048/480
        self.y_scale = param.y_scale #1536/480

        # change from screen coordinates to image coordinates
        self.x = param.x #boundaries[0][0][0]*self.x_scale
        self.x2 = param.x2 #boundaries[1][0][0]*self.x_scale
        self.y = param.y #boundaries[0][0][1]*self.y_scale
        self.y2 = param.y2 #boundaries[1][0][1]*self.y_scale

    def run(self):
        """Main tracking loop"""
        self._running = True
        self.lost_frames = 0
        self.frame_count = 0
        dataStorage = []
        self.iterNum = 0

        while self._running:
            data = self.que.get_latest(timeout=0.01)
            if data is not None:
                # try:
                
                img, timestamp = data
                if img is not None:
                    ret, data_ = self.process_frame(img)
                    if ret:
                        dataStorage.append((data_["position"][0], data_["position"][1], data_["radius"], timestamp))
                        # print("tracker got latest frame - queue size:", self.que.qsize(), timestamp)
                        self.frame_count += 1
                    else:
                        print("Tracking failed for current frame")
            else:
                time.sleep(0.001)
                QCoreApplication.processEvents()

        if self.save:
            np.save("./results/tracking_data.npy", np.stack(dataStorage, axis=0))
        
        self.init_all()

        print("Tracker thread is quited...")

    def process_frame(self, img):
        
        x_min = min(int(self.x), int(self.x2))
        x_max = max(int(self.x), int(self.x2))
        y_min = min(int(self.y), int(self.y2))
        y_max = max(int(self.y), int(self.y2))
        
        # Apply bounds checking
        roi_x1 = max(0, x_min)
        roi_y1 = max(0, y_min)
        roi_x2 = min(img.shape[1], x_max)
        roi_y2 = min(img.shape[0], y_max)

        # print("tracking frame:", roi_x1, roi_y1, roi_x2, roi_y2, img.shape)
        self.frameCrop = img[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if self.frameCrop.size == 0:
            print("Empty ROI - tracking lost")
            return False, []
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.frameCrop, (3, 3), 0)
        
        # Adaptive thresholding 
        #binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if binary is None:
            print("Otsu's thresholding failed")
            self.drop_frames += 1
            return False, []
        
        import os
        if os.path.exists('binary_image_1.png'):
            cv2.imwrite('binary_image_1.png', binary)

        binary = 255 - binary  # Invert if needed
        self.binary = binary.copy()

        if os.path.exists('binary_image_2.png'):  
            cv2.imwrite('binary_image_2.png', binary)
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.handle_tracking_lost()
            return False, []
        
        # Filter contours by area to remove noise
        max_area = self.frameCrop.size * 0.95  # Maximum 80% of ROI

        valid_contours = [c for c in contours if self.min_area < cv2.contourArea(c) < max_area]
        
        if not valid_contours:
            self.handle_tracking_lost()
            return False, []
        
        biggest = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        (x_, y_), radius = cv2.minEnclosingCircle(biggest)
        
        # Calculate center using moments (more accurate)
        M = cv2.moments(biggest)
        if M["m00"] == 0:
            self.handle_tracking_lost()
            return False, []
        
        cX = M["m10"] / M["m00"] #- 10*self.iterNum # X coordinate
        cY = M["m01"] / M["m00"] #+ 10*self.iterNum # Y coordinate
        self.iterNum += 1

        abs_x = roi_x1 + cX
        abs_y = roi_y1 + cY
        
        smoothing_factor = 0.1  # Adjust for more/less smoothing
        
        current_center_x = (self.x + self.x2) / 2
        current_center_y = (self.y + self.y2) / 2
        
        new_center_x = smoothing_factor * current_center_x + (1 - smoothing_factor) * abs_x
        new_center_y = smoothing_factor * current_center_y + (1 - smoothing_factor) * abs_y
        
        roi_width = self.x2 - self.x
        roi_height = self.y2 - self.y
        
        self.x = new_center_x - roi_width / 2
        self.x2 = new_center_x + roi_width / 2
        self.y = new_center_y - roi_height / 2
        self.y2 = new_center_y + roi_height / 2
        
        self.validateCoordinates()
        
        # Store tracking results
        center_x = (self.x + self.x2) / 2
        center_y = (self.y + self.y2) / 2
        
        # Reset lost frame counter on successful tracking
        self.lost_frames = 0
        
        # Emit tracking data
        tracking_data = {
            'position': (center_x, center_y),
            'area': area,
            'radius': radius,
            'frame_count': self.frame_count,
            'roi': (self.x, self.y, self.x2, self.y2),
            'lost': False
        }

        self.tracking_data.emit(tracking_data)
        # print(f"Tracked emitting: ({center_x:.2f}, {center_y:.2f}), Area: {area:.2f}, Radius: {radius:.2f}")
        return True, tracking_data
            
    
    def handle_tracking_lost(self):
        """Handle case when tracking is lost"""
        self.lost_frames += 1
        self.tracking_quality = 0.0
        
        if self.lost_frames >= self.max_lost_frames:
            
            self.tracking_data.emit({
                'position': None,
                'lost': True
            })
    
    def validateCoordinates(self):
        """Ensure coordinates stay within image bounds"""
        self.x = max(0, min(self.x, self.w - (self.x2 - self.x)))
        self.y = max(0, min(self.y, self.h - (self.y2 - self.y)))
        self.x2 = max(self.x + 10, min(self.x2, self.w))
        self.y2 = max(self.y + 10, min(self.y2, self.h))
  


    @pyqtSlot(Task)
    def receive_frame(self, data):
        """Update tracking parameters"""

        if data.id == "1000":
            print("Tracker received stop command")
            self._running = False
        elif data.id == "flush":
            print("Tracker received flush command - clearing queue")
            # Clear the processing queue
            self.que.clear()

        elif data.id == "1001":
            self.que.put(data.data)  # Add image to latest-only buffer
            
    