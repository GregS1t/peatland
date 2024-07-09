import os
import datetime
import torch
import sys
import pynvml



"""
This script is used to estimate the carbon footprint of the training process.   

Author  : Grégory Sainton
Lab     : Observatoire de Paris / LERMA
Date    : 2024-06-15
Version : 1.0

"""


class CarbonFootprintCalculator:
    def __init__(self, device: torch.device):
        self.device = device
        pynvml.nvmlInit()

    def __del__(self):
        pynvml.nvmlShutdown()

    def get_gpu_power_usage(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU
        power_usage_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # in milliwatts
        return power_usage_mw / 1000.0  # convert to watts

    def get_gpu_name(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetName(handle)

    def get_cpu_power_usage(self):
        return 150  # Default value for CPU power consumption in watts

    def calculate(self, start: datetime.datetime):
        end = datetime.datetime.now()
        total_training_time = (end - start).total_seconds()

        if self.device.type == 'cuda':
            power_consumption = self.get_gpu_power_usage()
        else:
            power_consumption = self.get_cpu_power_usage()

        total_energy_kwh = (power_consumption / 1000) * (total_training_time / 3600)
        carbon_intensity = 0.32  # kg CO2 per kWh en France in 2024
        total_carbon_footprint = total_energy_kwh * carbon_intensity

        return end, total_energy_kwh, total_carbon_footprint


class CarbonFootprintLogger:
    def __init__(self, carbon_log_dir: str, carbon_log_file: str):
        self.carbon_log_dir = carbon_log_dir
        self.carbon_log_file = carbon_log_file

    def log_carbon_footprint(self, end: datetime.datetime, total_energy_kwh: float, total_carbon_footprint: float):
        self._ensure_log_dir_exists()
        log_file_path = os.path.join(self.carbon_log_dir, self.carbon_log_file)
        mode = 'a' if os.path.exists(log_file_path) else 'w'

        with open(log_file_path, mode) as f:
            if mode == 'w':
                f.write("Time, Energy (kWh), Carbon footprint (kg CO2), Program\n")
            f.write(f"{end.strftime('%Y%m%d-%H%M%S')}, {total_energy_kwh:.4f}, {total_carbon_footprint:.4f}, {sys.argv[0]}\n")

    def _ensure_log_dir_exists(self):
        if not os.path.exists(self.carbon_log_dir):
            os.makedirs(self.carbon_log_dir)


def setup_device() -> torch.device:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    return device


if __name__ == "__main__":

    carbon_estimation = True
    carbon_log_dir = "/home/gsainton/CARBON_LOG"
    carbon_log_file = "carbon_footprint_test.log"


    if carbon_estimation:
        start = datetime.datetime.now()

    device = setup_device()
    print(f"Device: {device}   ")
    carbon_footprint_calculator = CarbonFootprintCalculator(device)
    carbon_logger = CarbonFootprintLogger(carbon_log_dir, carbon_log_file)

    end, total_energy_kwh, total_carbon_footprint = carbon_footprint_calculator.calculate(start)

    print("Total energy consumption (kWh): ", total_energy_kwh)

    carbon_logger.log_carbon_footprint(end, total_energy_kwh, total_carbon_footprint)
    print(f"GPU consumption: {carbon_footprint_calculator.get_gpu_power_usage()} W")
