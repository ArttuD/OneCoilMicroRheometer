"""
Configuration Loader
"""

import configparser
import os
import sys
from typing import Dict, List, Any, Tuple, Optional


class ConfigLoader:
    """Centralized configuration loader and manager"""
    
    def __init__(self, config_file):

        self.config = configparser.ConfigParser()
        
        # Determine config file path
        if config_file is None:
            # Use default config by default
            config_file = os.path.join(os.getcwd(), 'software/config', 'default_config.ini')
        elif not os.path.isabs(config_file):
            # If relative path, make it absolute from configs directory
            config_file = os.path.join(os.getcwd(), 'software/config', config_file)
            
        self.config_file = config_file
        
        # Load configuration
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        self.config.read(config_file)
        print(f"Loaded configuration from: {config_file}")
        
        # Cache parsed configurations for performance
        self._multi_choices_cache = None
        self._plot_dict_cache = None
        self._colors_list_cache = None
    
    def get_hardware_params(self):
        """Get hardware parameters (original default.ini parameters)"""

        section = self.config['DEFAULT']
        return {
            'buffer_size_cfg': section.getint('buffer_size_cfg'),
            'chans_in': section.getint('chans_in'),
            'time': section.getint('time'),
            'exposure': section.getint('exposure'),
            'framerate': section.getint('framerate'),
            'FirstResis': section.getfloat('FirstResis'),
            'samplingFreq': section.getint('samplingFreq'),
            'conversionFactor': section.getfloat('conversionFactor'),
            'save_root': section.get('save_root')
        }
    
    def get_ui_params(self) :
        """Get UI scaling and display parameters"""
        section = self.config['UI']
        return {
            'scaler': section.getint('scaler'),
            'max_image_width': section.getint('max_image_width'),
            'max_image_height': section.getint('max_image_height')
        }
    
    def get_colors_list(self):
        """Get color scheme for plots (replacement for hardcoded colors_list)"""
        if self._colors_list_cache is None:
            colors_str = self.config['UI'].get('colors')
            self._colors_list_cache = [color.strip() for color in colors_str.split(',')]
        return self._colors_list_cache
    
    def get_camera_params(self):
        """Get camera and optical system parameters"""
        section = self.config['CAMERA']
        return {
            'pixel_size': section.getfloat('pixel_size'),
            'objective': section.getfloat('objective'),
            'light_path_zoom': section.getfloat('light_path_zoom'),
            'camera_adapter_zoom': section.getfloat('camera_adapter_zoom'),
            'camera_height': section.getint('camera_height'),
            'camera_width': section.getint('camera_width'),
            'default_exposure': section.getint('default_exposure'),
            'default_framerate': section.getint('default_framerate'),
            'default_gain': section.getfloat('default_gain')
        }
    
    def get_nidevice_params(self):
        """Get NI device signal processing and control parameters"""
        section = self.config['NIDEVICE']
        return {
            'offset': section.getfloat('offset'),
            'Mg_offset': section.getfloat('Mg_offset'),
            'Mg_coef': section.getfloat('Mg_coef'),
            'grad': section.getfloat('grad'),
            'freq': section.getfloat('freq'),
            'start_time': section.getfloat('start_time'),
            'end_time': section.getfloat('end_time'),
            'totalTime': section.getfloat('totalTime'),
            'peak_distance': section.getfloat('peak_distance'),
            'n_peaks': section.getint('n_peaks'),
            # Filter settings
            'ema_filter_I_alpha': section.getfloat('ema_filter_I_alpha'),
            'ema_filter_B_alpha': section.getfloat('ema_filter_B_alpha'),
            # PI Controller settings
            'pi_controller_kp': section.getfloat('pi_controller_kp'),
            'pi_controller_ki': section.getfloat('pi_controller_ki'),
            'pi_controller_output_min': section.getfloat('pi_controller_output_min'),
            'pi_controller_output_max': section.getfloat('pi_controller_output_max')
        }
    
    def get_model_params(self):
        """Get simulation model parameters"""
        section = self.config['MODEL']
        return {
            'simulation_data_root': section.get('simulation_data_root'),
            'simulation_filter_sigma': section.getfloat('simulation_filter_sigma')
        }
    
    def get_multi_choices(self):
        """Get multi-choice dropdown options (replacement for hardcoded multi_choices)"""
        if self._multi_choices_cache is None:
            section = self.config['CHOICES']
            self._multi_choices_cache = {
                'feedback_mode': [item.strip() for item in section.get('feedback_mode').split(',')],
                'test_mode': [item.strip() for item in section.get('test_mode').split(',')],
                'waveform': [item.strip() for item in section.get('waveform').split(',')]
            }
        return self._multi_choices_cache
    
    def get_plot_dict(self):
        """Get plot configuration (replacement for hardcoded plot_dict)"""
        if self._plot_dict_cache is None:
            section = self.config['PLOTS']
            
            self._plot_dict_cache = {
                'currents': {
                    'ylabel': section.get('currents_ylabel'),
                    'yunits': section.get('currents_yunits'),
                    'xlabel': section.get('currents_xlabel'),
                    'xunits': section.get('currents_xunits'),
                    'title': section.get('currents_title'),
                    'y_limits': [section.getfloat('currents_y_min'), section.getfloat('currents_y_max')],
                    'curves': [curve.strip() for curve in section.get('currents_curves').split(',')]
                },
                'magnetic_field': {
                    'ylabel': section.get('magnetic_field_ylabel'),
                    'yunits': section.get('magnetic_field_yunits'),
                    'xlabel': section.get('magnetic_field_xlabel'),
                    'xunits': section.get('magnetic_field_xunits'),
                    'title': section.get('magnetic_field_title'),
                    'y_limits': [section.getfloat('magnetic_field_y_min'), section.getfloat('magnetic_field_y_max')],
                    'curves': [curve.strip() for curve in section.get('magnetic_field_curves').split(',')]
                },
                'track': {
                    'ylabel': section.get('track_ylabel'),
                    'yunits': section.get('track_yunits'),
                    'xlabel': section.get('track_xlabel'),
                    'xunits': section.get('track_xunits'),
                    'title': section.get('track_title'),
                    'y_limits': [section.getfloat('track_y_min'), section.getfloat('track_y_max')],
                    'curves': [curve.strip() for curve in section.get('track_curves').split(',')]
                }
            }
        return self._plot_dict_cache
    
    def get_ui_defaults(self) :
        """Get default values for UI configuration elements"""
        section = self.config['UI_DEFAULTS']
        return {
            'offset': section.get('offset_default'),
            'grad': section.get('grad_default'),
            'freq': section.get('freq_default'),
            'start_time': section.get('start_time_default'),
            'end_time': section.get('end_time_default'),
            'total_time': section.get('total_time_default'),
            'peak_distance': section.get('peak_distance_default'),
            'n_peaks': section.get('n_peaks_default'),
            'FrameRate (fps)': section.get('framerate_default'),
            'exposureTime (ms)': section.get('exposure_default'),
            'gain': section.get('gain_default'),
            'pixel_size': section.get('pixel_size_default'),
            'objective': section.get('objective_default'),
            'light_path_zoom': section.get('light_path_zoom_default'),
            'camera_adapter_zoom': section.get('camera_adapter_zoom_default'),
            'camera_height': section.get('camera_height_default'),
            'camera_width': section.get('camera_width_default')
        }
    
    def get_label_dimensions(self):
        ui_params = self.get_ui_params()
        scaler = ui_params['scaler']
        width = ui_params['max_image_width']
        height = ui_params['max_image_height']
        return [int(width/scaler), int(height/scaler)]
    
    def validate_config(self):
        required_sections = ['DEFAULT', 'UI', 'CAMERA', 'NIDEVICE', 'MODEL', 'CHOICES', 'PLOTS', 'UI_DEFAULTS']
        
        for section in required_sections:
            if section not in self.config:
                print(f"Missing required configuration section: {section}")
                return False
        
        print("Configuration validation passed")
        return True


# Global configuration instance - initialized when module is imported
_config_instance: Optional[ConfigLoader] = None

def get_config(config_file: Optional[str] = None):
    """
    Get global configuration 
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigLoader(config_file)
        if not _config_instance.validate_config():
            raise ValueError("Configuration validation failed")
    
    return _config_instance

def reset_config():
    global _config_instance
    _config_instance = None


# Convenience functions for backward compatibility
def get_multi_choices() -> Dict[str, List[str]]:
    """Get multi-choice options (replacement for hardcoded multi_choices)"""
    return get_config().get_multi_choices()

def get_plot_dict() -> Dict[str, Dict[str, Any]]:
    """Get plot configuration (replacement for hardcoded plot_dict)"""
    return get_config().get_plot_dict()

def get_colors_list() -> List[str]:
    """Get color list (replacement for hardcoded colors_list)"""
    return get_config().get_colors_list()


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = get_config()
        print("Hardware params:", config.get_hardware_params())
        print("Multi choices:", config.get_multi_choices())
        print("Colors:", config.get_colors_list())
        print("Plot dict keys:", list(config.get_plot_dict().keys()))
        print("Configuration loading test passed!")
    except Exception as e:
        print(f"Configuration loading test failed: {e}")