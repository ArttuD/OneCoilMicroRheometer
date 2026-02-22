#!/usr/bin/env python3
"""
Microrheology Controller Application
Author: Arttu Lehtonen
Version: 2.0
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

import configparser

from PyQt6.QtWidgets import QApplication
from tools.pyqt import MainWindow
from tools.config import get_config

import traceback


def parse_arguments():
    """
    Parse command line arguments with comprehensive help and validation.
    """
    parser = argparse.ArgumentParser(
        description='Professional Microrheology Test Setup Controller',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--path', '-p', 
        type=str, 
        help='Save path for measurement files (will be created if it does not exist)'
    )
    parser.add_argument(
        '--user', '-u', 
        default='DEFAULT',
        help='Configuration profile to use'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Specific configuration file to use (overrides default unified config)'
    )
    parser.add_argument(
        '-d', '--debug',
        help='Enable debug logging',
        action='store_const', 
        dest='loglevel',
        const=logging.DEBUG,
        default=logging.WARNING
    )
    parser.add_argument(
        '-v', '--verbose',
        help='Enable verbose logging',
        action='store_const', 
        dest='loglevel',
        const=logging.INFO
    )
    
    return parser.parse_args()


def load_configuration(config_file, logger=None):
    """
    Load application configuration from file.
    """
    try:
        if config_file and config_file.endswith('.ini'):

            # Legacy configuration support
            config = configparser.ConfigParser()
            config_path = Path.cwd() / 'configs' / config_file
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
                
            config.read(config_path)
            unified_config = None
            if logger:
                logger.info(f"Loaded legacy configuration: {config_path}")
            
        else:
            # Use new unified configuration system
            unified_config = get_config(config_file)
            config = configparser.ConfigParser()
            hardware_params = unified_config.get_hardware_params()
            config['DEFAULT'] = {str(k): str(v) for k, v in hardware_params.items()}
            if logger:
                logger.info("Loaded unified configuration system")
            
        return config, unified_config
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to load configuration: {e}")
        else:
            print(f"Configuration error: {e}")
        raise


def create_save_directory(path: str, logger=None) -> None:
    """
    Create save directory if it doesn't exist.
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        if logger:
            logger.info(f"Save directory ready: {path}")

    except OSError as e:
        if logger:
            logger.error(f"Failed to create save directory {path}: {e}")
        raise

def main() -> int:
    """
    Main application
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger = None
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration
        config, unified_config = load_configuration(args.config)
        # Set default save path if not provided
        if not args.path:
            args.path = config["DEFAULT"]["save_root"]
            
        # Parse configuration parameters with proper error handling
        try:
            args.buffer_size_cfg = int(config[args.user]["buffer_size_cfg"])
            args.chans_in = int(config[args.user]["chans_in"])
            args.time = int(config[args.user]["time"])
            args.exposure = int(config[args.user]["exposure"])
            args.framerate = int(config[args.user]["framerate"])
            args.FirstResis = float(config[args.user]["FirstResis"])
            args.samplingFreq = int(config[args.user]["samplingFreq"])
            args.conversionFactor = float(config[args.user]["conversionFactor"])
        except (KeyError, ValueError) as e:
            return 1
            
        # Create save directory
        create_save_directory(args.path)
        app = QApplication(sys.argv)
        app.setApplicationName("Microrheology Controller")
        app.setApplicationVersion("2.0")
        app.setOrganizationName("Your Organization")
        
        # Create main window
        main_window = MainWindow(args)
        main_window.show()
        
        # Run application event loop
        exit_code = app.exec()
        return exit_code
        
    except KeyboardInterrupt:
        msg = "Application interrupted by user (Ctrl+C)"
        print(msg)
        return 0
    except Exception as e:
        msg = f"Fatal error: {e}"
        print(f"ERROR: {msg}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())