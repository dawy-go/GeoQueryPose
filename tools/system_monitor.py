import math
import os
from numbers import Number


class NvitopSystemMonitor:
    """Write GPU metrics to TensorBoard."""

    def __init__(self, enabled=True, logger=None):
        self.enabled = bool(enabled)
        self.logger = logger
        self._warned = False
        self._devices = None
        self._host_process = None
        self._nvitop_na = None

        if self.enabled:
            self._init_nvitop()

    def record(self, writer, step):
        if not self.enabled or writer is None:
            return

        try:
            metrics = self.collect()
        except Exception as exc:
            self._warn_once(f"nvitop system monitor failed and will be disabled: {exc}")
            self.enabled = False
            return

        for name, value in metrics.items():
            writer.add_scalar(f"system/{name}", value, step)

    def collect(self):
        if not self.enabled:
            return {}

        if self._devices is None:
            self._init_nvitop()
        if not self.enabled:
            return {}

        metrics = {}
        gpu_utils = []
        memory_used_total = 0.0
        memory_total_total = 0.0

        for device in self._devices:
            index = self._device_index(device)
            prefix = f"gpu_{index}"

            gpu_util = self._read_number(device, "gpu_utilization")
            memory_util = self._read_number(device, "memory_utilization")
            memory_used = self._read_number(device, "memory_used")
            memory_total = self._read_number(device, "memory_total")
            memory_percent = self._read_number(device, "memory_percent")
            temperature = self._read_number(device, "temperature")
            encoder_util = self._read_number(device, "encoder_utilization")
            decoder_util = self._read_number(device, "decoder_utilization")

            self._add(metrics, f"{prefix}_gpu_utilization_percent", gpu_util)
            self._add(metrics, f"{prefix}_memory_utilization_percent", memory_util)
            self._add(metrics, f"{prefix}_memory_used_mib", self._bytes_to_mib(memory_used))
            self._add(metrics, f"{prefix}_memory_percent", memory_percent)
            self._add(metrics, f"{prefix}_temperature_c", temperature)
            self._add(metrics, f"{prefix}_encoder_utilization_percent", encoder_util)
            self._add(metrics, f"{prefix}_decoder_utilization_percent", decoder_util)

            if gpu_util is not None:
                gpu_utils.append(gpu_util)
            if memory_used is not None:
                memory_used_total += memory_used
            if memory_total is not None:
                memory_total_total += memory_total

        if gpu_utils:
            metrics["gpu_utilization_percent_mean"] = sum(gpu_utils) / len(gpu_utils)
        if memory_total_total > 0:
            metrics["memory_used_mib_total"] = self._bytes_to_mib(memory_used_total)
            metrics["memory_total_mib_total"] = self._bytes_to_mib(memory_total_total)
            metrics["memory_percent_total"] = memory_used_total / memory_total_total * 100.0

        return metrics

    def _init_nvitop(self):
        try:
            from nvitop import Device, HostProcess, NA

            self._nvitop_na = NA
            self._devices = list(Device.cuda.all())
            self._host_process = HostProcess(os.getpid())
        except Exception as exc:
            self._warn_once(f"nvitop system monitor is disabled: {exc}")
            self.enabled = False
            self._devices = []

    def _collect_current_process(self):
        metrics = {}
        process = self._host_process
        if process is None:
            return metrics

        cpu_percent = self._read_number(process, "cpu_percent")
        memory_percent = self._read_number(process, "memory_percent")
        running_time = self._read_number(process, "running_time_in_seconds")
        host_memory = self._read_number(process, "host_memory")

        self._add(metrics, "process_cpu_percent", cpu_percent)
        self._add(metrics, "process_memory_percent", memory_percent)
        self._add(metrics, "process_running_time_sec", running_time)
        self._add(metrics, "process_host_memory_mib", self._bytes_to_mib(host_memory))
        return metrics

    def _read_number(self, obj, name):
        attr = getattr(obj, name, None)
        if attr is None:
            return None
        try:
            value = attr() if callable(attr) else attr
        except Exception:
            return None
        return self._to_float(value)

    def _to_float(self, value):
        if value is None or value is self._nvitop_na:
            return None
        if isinstance(value, Number):
            value = float(value)
            return value if math.isfinite(value) else None
        return None

    def _device_index(self, device):
        for name in ("cuda_index", "index", "nvml_index"):
            value = self._read_number(device, name)
            if value is not None:
                return int(value)
        return len(str(device))

    @staticmethod
    def _bytes_to_mib(value):
        return None if value is None else value / (1024.0 * 1024.0)

    @staticmethod
    def _milliwatts_to_watts(value):
        return None if value is None else value / 1000.0

    @staticmethod
    def _add(metrics, name, value):
        if value is not None:
            metrics[name] = value

    def _warn_once(self, message):
        if self._warned:
            return
        self._warned = True
        if self.logger is not None:
            self.logger.warning(message)
