
"""
National Instruments DAQ Device Controller
Author: Arttu Lehtonen
Version: 2.0
"""

import datetime
import os
import time
from pathlib import Path
import queue
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nidaqmx
import nidaqmx.constants
import numpy as np
from nidaqmx.stream_readers import AnalogMultiChannelReader
from PyQt6.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication

from tools.config import get_config
from tools.tools import EMAFilter, PIController

class NiDevice(QObject):
    """
    Professional National Instruments DAQ device controller.
    
    Signals:
        NI_setData: Emits processed measurement data
        NI_print_str: Emits status and diagnostic messages
    """
    
    # Qt Signals
    NI_setData = pyqtSignal(object)
    NI_print_str = pyqtSignal(str)

    def __init__(self, args: Any) -> None:
        """
        Initialize NI-DAQ device controller.
        
        Args:
            args: Command line arguments containing hardware configuration
            
        Raises:
            ConnectionError: If DAQ device cannot be initialized
            ValueError: If configuration parameters are invalid
        """
        super().__init__()
        
        self.args = args
        self.root = Path(args.path)
        
        # Ensure save directory exists
        self.root.mkdir(parents=True, exist_ok=True)

        self.Flag_mode = None
        self.Flag_save = None
        self.Flag_feedBack = None
        self.wavetype = None

        self.mode = "0"

        self.current_sequence = None
        self.res_1 = 0.0    
        self.cur_1 = 0.0

        try:
            # Load NI device parameters from configuration
            config = get_config()
            ni_params = config.get_nidevice_params()
            
            # Signal processing parameters
            self.offset = ni_params['offset']
            self.Mg_offset = ni_params['Mg_offset']
            self.Mg_coef = ni_params['Mg_coef']
            self.grad = ni_params['grad']
            self.freq = ni_params['freq']
            self.start_time = ni_params['start_time']
            self.end_time = ni_params['end_time']
            self.totalTime = ni_params['totalTime']
            self.peak_distance = ni_params['peak_distance']
            self.n_peaks = ni_params['n_peaks']
            
            # Store config parameters for filter initialization
            self._ni_config = ni_params
            
        except Exception as e:
            raise ValueError(f"Invalid NI device configuration: {e}")
        
        self._running = False
        self.iteration = 0
        self.dropped_samples = 0

        # Track if data acquisition completed successfully
        self._data_completed = False

        #parse arguments
        self.totalTime = args.time
        self.buffer_size = round(args.buffer_size_cfg)
        self.samplingFreq = args.samplingFreq
        self.chans_in = args.chans_in

        # Calculate derived parameters and reserve memory
        self.NSamples = int(self.samplingFreq * self.totalTime / self.buffer_size)
        self.FreqRet = int(self.samplingFreq / self.buffer_size)
        
        # Calibration parameters
        self.B2V_voltage = float(args.conversionFactor)
        
        # Initialize data structures
        self.init_buffers_queues()
        
        # Hardware connection status
        try:
            self.connected = True
            
        except Exception as e:
            error_msg = f"NI device initialization error: {str(e)}"
            self.NI_print_str.emit(error_msg)
            self.connected = False
            raise ConnectionError(f"NI-DAQ initialization failed: {e}")

        

    def _reset_variables(self):

        self._running = False
        self.iteration = 0
        self.dropped_samples = 0
        self.mode = "0"
        self._data_completed = False

        # Clear all internal queues first
        self._clear_all_queues()
        
        # Reinitialize clean queues
        self.init_buffers_queues()
        self._init_filters()
        
        # Process any pending Qt events to clear signal/slot queues
        QCoreApplication.processEvents()
        
    def _clear_all_queues(self):
        """Clear all internal queues to remove pending messages"""
        # Clear existing queues if they exist
        if hasattr(self, 'data_queue'):
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except:
                    break
                    
        if hasattr(self, 'writeQue'):
            while not self.writeQue.empty():
                try:
                    self.writeQue.get_nowait()
                except:
                    break
                    
        if hasattr(self, 'resOneQue'):
            while not self.resOneQue.empty():
                try:
                    self.resOneQue.get_nowait()
                except:
                    break

        if hasattr(self, 'model_queue'):
            while not self.model_queue.empty():
                try:
                    self.model_queue.get_nowait()
                except:
                    break

    def init_buffers_queues(self):
        # Separate queue for data handling (larger buffer)
        self.data_queue = Queue(maxsize=1000)
        self.writeQue = Queue(maxsize=100)
        self.resOneQue = Queue(maxsize=100)
        self.model_queue = Queue(maxsize=100)

        # Always use current NSamples value for buffer sizing
        current_NSamples = int(self.samplingFreq*self.totalTime/self.buffer_size)
        self.NpyStorage = np.zeros((6, current_NSamples))
        self.bufferIn = np.zeros((self.chans_in, self.buffer_size))
        self.values = np.zeros((self.chans_in,1))

    def _update_params(self, params):

        
        self.sync_time = params["timestamp_sync"]
        self.Flag_mode = params["Flag_mode"]
        self.Flag_save = params["Flag_save"]
        self.Flag_feedBack = params["Flag_feedBack"]

        self.current_sequence = params["current_sequence"]
        
        self.res_1 = params["res_1"]

        self.offset = params["offset"]
        self.grad = params["grad"]
        self.freq = params["freq"]
        self.start_time = params["start_time"]
        self.end_time = params["end_time"]
        self.totalTime = params["total_time"]
        self.peak_distance = params["peak_distance"]
        self.n_peaks = params["n_peaks"]

        self.NSamples = int(self.samplingFreq*self.totalTime/self.buffer_size)
        self.FreqRet = int(self.samplingFreq/self.args.buffer_size_cfg)


        self.init_buffers_queues()



    def _init_filters(self):
        # Use configuration parameters for filter initialization
        self.filter_I = EMAFilter(alpha=self._ni_config['ema_filter_I_alpha'])
        self.filter_B = EMAFilter(alpha=self._ni_config['ema_filter_B_alpha'])

        self.controller = PIController(
            kp=self._ni_config['pi_controller_kp'], 
            ki=self._ni_config['pi_controller_ki'], 
            setpoint=0,
            output_limits=[self._ni_config['pi_controller_output_min'], self._ni_config['pi_controller_output_max']],
            sample_time=1/self.FreqRet
        )

    def _close_task(self):
        self._running = False
        for i in [self.NiDWriter, self.NiAlWriter, self.NiAlReader]:
            try:
                i.stop()
            except Exception as e:
                print(f"Error stopping task: {str(e)}")
            try:
                i.close()
            except Exception as e:
                print(f"Error closing task: {str(e)}")

    def _init_task(self):
        """
        Init and cfg tasks
        """
        self.NiAlReader = nidaqmx.Task()
        self.NiAlWriter = nidaqmx.Task()
        self.NiDWriter = nidaqmx.Task()

        self.cfg_AO_writer_task()
        self.cfg_DO_writer_task()
        self.cfg_AL_reader_task()

    def cfg_AO_writer_task(self):
        #Config analog output channel between -10 and 10V
        self.NiAlWriter.ao_channels.add_ao_voltage_chan("Dev1/ao0","", min_val=- 10.0, max_val=10.0)

    def cfg_DO_writer_task(self):
        #Config digital output
        self.NiDWriter.do_channels.add_do_chan("Dev1/port1/line0","") #Dev1/port0/line0

    def cfg_AL_reader_task(self):
        """
        Config continous analog input channel
        """
        # Current reading differential 
        self.NiAlReader.ai_channels.add_ai_voltage_chan("Dev1/ai0", terminal_config = nidaqmx.constants.TerminalConfiguration.DIFF)
        # Mg sensor single ended
        self.NiAlReader.ai_channels.add_ai_voltage_chan("Dev1/ai1", terminal_config = nidaqmx.constants.TerminalConfiguration.RSE, min_val = 0, max_val=5)  
        # sampling rate
        self.NiAlReader.timing.cfg_samp_clk_timing(rate=self.samplingFreq, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                            samps_per_chan=self.buffer_size)
        #Data stream and buffer
        self.readerStreamIn = AnalogMultiChannelReader(self.NiAlReader.in_stream)
        self.NiAlReader.register_every_n_samples_acquired_into_buffer_event(self.buffer_size, self.readingCallback)

    def readingCallback(self, task_handle, event_type, number_of_samples, callback_data):
        """
        Fast callback - minimal processing, no file I/O
        """
        try:
            if not self._running or self.iteration >= self.NSamples:
                return 0
                
            # Read the actual number of samples available
            samples_to_read = min(number_of_samples, self.buffer_size)
            
            # Read data from DAQ
            self.readerStreamIn.read_many_sample(
                self.bufferIn, 
                number_of_samples_per_channel=samples_to_read,
                timeout=1.0
            )
            
            
            # Calculate current values (avoid memory growth)
            voltage_mean = np.mean(self.bufferIn[0, :samples_to_read])
            magnetic_mean = -1*((np.mean(self.bufferIn[1, :samples_to_read])-self.Mg_offset)/self.Mg_coef)
            # print(voltage_mean/0.12,  "/",self.res_1)
            # Create data packet with timestamp
            timestamp = time.time()
            data_packet = {
                'timestamp': timestamp,
                'iteration': self.iteration,
                'voltage_mean': voltage_mean,
                'magnetic_mean': magnetic_mean,
                "type": "data"
            }
            
            # Put in queue for data handling thread (non-blocking)
            if not self.data_queue.full():
                self.data_queue.put(data_packet)
            else:
                # Queue full - data loss warning
                self.dropped_samples += 1
            
            self.iteration += 1
            
            # Check completion
            if self.iteration >= self.NSamples:
                self._running = False
                self.data_queue.put({'type': 'stop'})  # Stop signal
                self.NiAlReader.stop()
                
        except Exception as e:
            self.NI_print_str.emit(f"Callback error: {str(e)}")
            self._running = False
            self.data_queue.put({'type': 'error', 'message': str(e)})
            self.NiAlReader.stop()
            
        return 0

    def data_handler_thread(self):            
        """
        Separate thread for data processing and file I/O
        """
        iteration = 0                    
        # Initialize with a safe default
        output_voltage = 0.0
        # Track if data acquisition completed successfully
        self._data_completed = False
        self.model_freedback = 0.0
        
        try:
            while self._running and iteration < self.NSamples:
                try:

                    # Get data from queue (blocking with timeout)
                    data = self.data_queue.get(timeout=0.5)  # Shorter timeout for responsiveness
                    # Quick emit for real-time display
                    
                    if isinstance(data, dict) and data.get('type') == 'stop':
                        break
                    elif isinstance(data, dict) and data.get('type') == 'error':
                        self.NI_print_str.emit(f"Data handler got error: {data.get('message')}")
                        break
                    elif iteration >= self.NSamples:
                        break

                    data["voltage_mean"] = self.filter_I.update(data["voltage_mean"])
                    data["magnetic_mean"] = self.filter_B.update(data["magnetic_mean"])
                    
                    if iteration % 30 == 0:
                        QCoreApplication.processEvents()

                    if not self.resOneQue.empty():
                        self.res_1 = self.resOneQue.get()

                    if not self.model_queue.empty():
                        self.model_freedback = self.model_queue.get_nowait()
                        # print("received model feedback:", self.cur_1, " / ", self.model_freedback)

                    if self.Flag_mode == "manual":
                        if not self.writeQue.empty():
                            self.cur_1 = self.writeQue.get_nowait()
                    else:
                        self.cur_1 = self.sequence[iteration]

                        if self.cur_1 == 0.0:
                            self.model_freedback= 0.0
                    

                        
                    target = ((self.cur_1) - self.model_freedback)#*1.1 #amplifier gain
                    
                    if self.Flag_feedBack == "voltage":
                        output_voltage = self.controller.update(data["voltage_mean"]/self.res_1, target)
                    elif self.Flag_feedBack == "magnetic_flux":
                        output_voltage = self.controller.update(data["magnetic_mean"]/self.B2V_voltage, (target+1e-3))
                    else:
                        self.NI_print_str.emit(f"Warning: Unknown feedback mode '{self.Flag_feedBack}', using 0V output")
                    # print("writing", output_voltage, "A or", self.cur_1, "A target\n Measured", data["voltage_mean"], "and", data["magnetic_mean"],)
                    try:
                        self.NiAlWriter.write(output_voltage,1000)
                    except Exception as e:  
                        self.NI_print_str.emit(f"Error writing output voltage: {str(e)}")
                        break
                    
                    data_packet = {
                            'timestamp': data["timestamp"]-self.sync_time,
                            'reader_iteration': data["iteration"],
                            'writer_iteration': iteration,
                            'current_target':  self.cur_1,
                            'current_measured': data["voltage_mean"]/self.res_1,
                            'magnetic_field':  data["magnetic_mean"],
                            "type": "data"
                            } #

                    if iteration < self.NSamples:
                        self.NpyStorage[0, iteration] = data_packet['timestamp']
                        self.NpyStorage[1, iteration] = data_packet['current_target']
                        self.NpyStorage[2, iteration] = data_packet['current_measured']
                        self.NpyStorage[3, iteration] = data_packet['magnetic_field']
                        self.NpyStorage[4, iteration] = iteration
                        self.NpyStorage[5, iteration] = self.model_freedback
    
                    # Put in legacy queue for backward compatibility
                    if iteration%30 == 0:
                        self.NI_setData.emit(data_packet)

                    iteration += 1

                except queue.Empty:
                    # Timeout on queue get - just continue to check _running flag
                    continue
                except Exception as e:
                    self.NI_print_str.emit(f"Data handler error: {str(e)}")
                    continue
            
            # Mark as completed if we processed all expected samples
            if iteration >= self.NSamples:
                self._data_completed = True
                
        finally:
            self._running = False
            self.write_empty()
            
            data_packet = {
                    'timestamp': data["timestamp"]-self.sync_time,
                    'reader_iteration': data["iteration"],
                    'writer_iteration': iteration,
                    'current_target':  0,
                    'current_measured': 0,
                    'magnetic_field':  0,
                    "type": "data"
                }
            self.NI_setData.emit(data_packet)

            if self.Flag_save:
                np.save(os.path.join(self.root,"driver.npy"), self.NpyStorage)
            
            self.NI_print_str.emit(f"Data handler stopped. Dropped samples: {self.dropped_samples}")
            QCoreApplication.processEvents()

    def write_empty(self):

        try:
            for i in range(5):
                self.NiAlWriter.write(0,1000)
        except Exception as e:
            self.NI_print_str.emit(f"Error writing zero voltage: {str(e)}")

    def close(self):
        self._running = False
        for i in [self.NiDWriter, self.NiAlWriter, self.NiAlReader]:
            try:
                i.stop()
            except Exception as e:
                self.NI_print_str.emit(f"Error stopping task: {str(e)}")
            try:
                i.close()
            except Exception as e:
                self.NI_print_str.emit(f"Error closing task: {str(e)}")

    def measure(self):

        # Reset completion flag at start of measurement
        self._data_completed = False
        
        self._init_task()
        self._init_filters()

        self.NSamples = int(self.samplingFreq*self.totalTime/self.buffer_size)
        # print(f"NSamples: {self.NSamples}, totalTime: {self.totalTime}, buffer_size: {self.buffer_size}")
        
        if self.current_sequence == "slope":
            self.generateSlope()
        elif self.current_sequence == "step":
            self.generateStepWave()
        elif self.current_sequence == "pulses":
            self.generatePulses()
        elif self.current_sequence == "sine":
            self.SinWave()
        
        # print(f"Generated sequence shape: {self.sequence.shape}, mode was {self.current_sequence}")

        self.NiDWriter.start()
        self.NiDWriter.write(True, 1000)
        self.NiAlWriter.start()
        self.write_empty()

        self.NiAlReader.start() #start reading
        self.data_handler_thread()
        
        return 1
    
    def calibrate(self):

        # Reset completion flag at start of calibration
        self._data_completed = False
        
        self._init_task()
        self._init_filters()

        self.totalTime = 30
        self.NSamples = int(self.samplingFreq*self.totalTime/self.buffer_size)
        self.generateSlope()

        self.NiDWriter.start()
        self.NiDWriter.write(True, 1000)
        self.NiAlWriter.start()
        self.write_empty()

        self.NiAlReader.start() #start reading

        self.data_handler_thread()

        # Only attempt fitting if calibration sequence completed successfully
        if getattr(self, '_data_completed', False):
            try:
                self.fitCalibration()
                self.NI_print_str.emit("Calibration completed.")
            except Exception as e:
                self.NI_print_str.emit(f"Calibration fitting error: {str(e)}")
        else:
            self.NI_print_str.emit("Calibration was interrupted - conversion factor not updated")
        
        return 1

    def fitCalibration(self):
        
        # Simple check to prevent crash if no data points found
        
        indices = np.where(self.NpyStorage[1,:] >= 0.75)[0]

        if len(indices) == 0:
            print("No calibration data found with current >= 0.75")
            return
        
        maxIndex = indices[0]

        x = self.NpyStorage[2,self.cutOFF:maxIndex]#*self.res_1
        y = self.NpyStorage[3,self.cutOFF:maxIndex]
        # print("found", x, y)
        k, b= np.polyfit(x,y,1)
        
        self.B2V_voltage = float(k)
        # print("found", self.B2V_voltage)
        plt.scatter(x,y, label = "data", color = "red", alpha= 0.4)
        plt.plot(x, k*x +b, label = "fit", color = "blue")
        plt.legend()
        # print("saving figure...")
        plt.savefig(os.path.join(self.root,'calib_{}.png'.format(datetime.date.today())))   # save the figure to file
        # print("figure saved")
        plt.close()

        # print("saved in {}".format(os.path.join(self.root,'calib_{}.png'.format(datetime.date.today()))))
        self.NI_print_str.emit("fit results:  k: {}, b: {}\nSaved in {}".format(k, b, os.path.join(self.root,'calib_{}.png'.format(datetime.date.today()))))
       

    def run(self):
        """Main worker loop:
        Modes:
            "0" - idle
            "1" - calibration
            "2" - measurement
        """
        # print("NI worker thread started")
        # persistent loop
        if self.mode == "1":
            # Calibration
            self.NI_print_str.emit("Starting calibration mode")
            self._running = True
            self.Flag_mode = "sequence"
            # default to voltage feedback unless overridden
            if not self.Flag_feedBack:
                self.Flag_feedBack = "voltage"
            try:
                self.calibrate()
            except Exception as e:
                self.NI_print_str.emit(f"Calibration error: {e}")

            # Signal completion and reset state
            self.NI_print_str.emit("Calibration completed.")
            try:
                self.NI_print_str.emit(f"Value: {self.B2V_voltage}")
            except Exception:
                pass

            self._running = False
            # reset mode to idle
            self.mode = "0"
            # reinitialize buffers for next run
            self.init_buffers_queues()
            self._close_task()

        elif self.mode == "2":
            # Measurement
            self._running = True
            
            try:
                # measure() runs data handling and will return when done or stopped
                self.measure()
            except Exception as e:
                self.NI_print_str.emit(f"Measurement error: {e}")

            self._running = False
            self.mode = "0"
            self.init_buffers_queues()
            self._close_task()

        else:
            # idle - small sleep to avoid busy loop
            time.sleep(0.05)

        QCoreApplication.processEvents()

        

    @pyqtSlot(object)
    def receive_main(self,task):
        # print("received task:", task)
        # Use task.data (Task class uses .data) and support stop/interrupt commands.
        try:
            tid = task.id
            tdata = getattr(task, 'data', None)
        except Exception:
            return

        if tid == "1001":
            # write queue (e.g., set manual write value)
            self.writeQue.put(tdata)
        elif tid == "1002":
            self.resOneQue.put(tdata)
        elif tid == "1003":
            # print("received model feedback:", tdata)
            self.model_queue.put(tdata["d_i_ref"])
        elif tid == "1000":
            print(f"NI received stop command (current mode: {self.mode}, running: {self._running})")
            # Process stop command and do thorough cleanup
            if self.mode in ["1", "2"] and self._running:
                print("Processing stop command - stopping operation and clearing queues")
                self._running = False
                # also set mode to idle so the run loop won't immediately restart
                self.mode = "0"
                # Clear all queues to prevent any leftover messages
                self._clear_all_queues()
                # Process events to clear signal queue
                QCoreApplication.processEvents()
            else:
                print("Ignoring stop command - not in active measurement/calibration mode")
        elif tid == "flush":
            # Flush command to clear all pending messages
            print("NI received flush command - clearing all queues")
            self._clear_all_queues()
            QCoreApplication.processEvents()
        else:
            # Allow other commands to set the mode directly by sending a task with id 'mode'
            if tid == "mode":
                # expect tdata to be a string like "1" or "2"
                self.mode = str(tdata)
            # ignore unknown ids

    def generateStepWave(self):
        """
        Generate step sequence
        """
        

        self.sequence = np.zeros(self.NSamples)

        start = int(self.start_time*self.FreqRet)
        end = int(self.end_time*self.FreqRet)

        print(f"Generating step wave: {start} to {end}, offset: {self.offset}")

        self.sequence[start:end] = self.offset

    def generatePulses(self):

        self.sequence = np.zeros(self.NSamples)

        slope = 0.2
        i_max = 1.2
        period = 2          # seconds
        spacing = 10        # seconds

        start_idx = int(1 * spacing * self.FreqRet)
        for idx, i in enumerate(range(2, int(1.5 * i_max / slope))):
            if start_idx >= self.NSamples:
                break

            offset = slope * i - 0.2
            # fixed pulse duration for all pulses
            dur_samps = int(period * self.FreqRet)            # CHANGED: remove scaling by i
            end_idx = min(start_idx + dur_samps, self.NSamples)
            if end_idx <= start_idx:
                break

            self.sequence[start_idx:end_idx] = offset

            # leave gap (no-moment) intact
            start_idx = end_idx + int(spacing * self.FreqRet)

    def generateSlope(self):
        """
        Generate linear slope for calibration
        """
        #print("Generating slope")
        self.sequence = np.zeros(self.NSamples)

        self.cutOFF = 2*self.FreqRet
        N_real_samples = self.NSamples - self.cutOFF

        offset = 0.1

        self.sequence[:int(self.cutOFF)] = offset
        self.sequence[int(self.cutOFF):] = np.linspace(offset, 1.2, N_real_samples)
        #self.sequence[-1] = 0

    def SinWave(self):
        """
        Generate sine wave sequence
        """
        # print("Generating sine wave")
        self.sequence = np.zeros(self.NSamples)
        
        # Ensure we don't exceed the sequence length
        start = int(self.start_time*self.FreqRet)
        end = int(self.end_time*self.FreqRet)

        if start < self.NSamples:
            self.sequence[start:end] = 0.5

        for i in range(len(np.arange(end, self.NSamples))):
            # Calculate the actual number of samples for the sine wave
            step_time = (i-end)/self.FreqRet
            self.sequence[i] = self.offset + self.grad*np.sin(2*np.pi*self.freq*step_time)
