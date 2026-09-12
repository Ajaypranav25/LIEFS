"""
Hardware profiling and benchmark scoring for LIEFS.

Provides automated hardware inspection (CPU, GPU, RAM, VRAM, OS) and computes
effective memory bandwidth and standardized computer benchmark scores.
"""

import platform
from dataclasses import asdict, dataclass

import psutil
import torch


@dataclass
class HardwareProfile:
    """Detailed hardware profile of the host computer."""
    # Platform & OS
    os_name: str
    os_release: str
    architecture: str
    python_version: str
    pytorch_version: str
    
    # CPU
    cpu_model: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    cpu_freq_mhz: float | None
    ram_total_gb: float
    ram_available_gb: float
    ram_usage_percent: float
    
    # GPU / Accelerator
    cuda_available: bool
    gpu_count: int
    gpu_name: str
    gpu_compute_capability: str | None
    vram_total_mb: float
    vram_free_mb: float
    vram_allocated_mb: float
    vram_reserved_mb: float
    vram_usage_percent: float
    driver_version: str | None = None
    cuda_version: str | None = None
    
    def to_dict(self) -> dict:
        return asdict(self)


def get_cpu_name() -> str:
    """Retrieve human-readable CPU brand name."""
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            return cpu_name.strip()
        elif platform.system() == "Darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_hardware_profile() -> HardwareProfile:
    """Detect and return full hardware specifications of the host computer."""
    # RAM
    ram = psutil.virtual_memory()
    ram_total_gb = round(ram.total / (1024 ** 3), 2)
    ram_available_gb = round(ram.available / (1024 ** 3), 2)
    ram_usage_percent = ram.percent
    
    # CPU
    cpu_freq = psutil.cpu_freq()
    freq_mhz = round(cpu_freq.current, 1) if cpu_freq else None
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1
    cpu_model = get_cpu_name()
    
    # GPU
    cuda_avail = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_avail else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail and gpu_count > 0 else "None (CPU Only)"
    
    compute_cap = None
    vram_total_mb = 0.0
    vram_free_mb = 0.0
    vram_allocated_mb = 0.0
    vram_reserved_mb = 0.0
    vram_usage_pct = 0.0
    cuda_ver = torch.version.cuda if cuda_avail else None
    
    if cuda_avail and gpu_count > 0:
        cap = torch.cuda.get_device_capability(0)
        compute_cap = f"{cap[0]}.{cap[1]}"
        props = torch.cuda.get_device_properties(0)
        vram_total_mb = round(props.total_memory / (1024 * 1024), 2)
        vram_allocated_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
        vram_reserved_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
        vram_free_mb = round(vram_total_mb - vram_allocated_mb, 2)
        vram_usage_pct = round((vram_allocated_mb / vram_total_mb * 100), 1) if vram_total_mb > 0 else 0.0
        
    return HardwareProfile(
        os_name=platform.system(),
        os_release=platform.release(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cpu_model=cpu_model,
        cpu_physical_cores=physical_cores,
        cpu_logical_cores=logical_cores,
        cpu_freq_mhz=freq_mhz,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        ram_usage_percent=ram_usage_percent,
        cuda_available=cuda_avail,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_compute_capability=compute_cap,
        vram_total_mb=vram_total_mb,
        vram_free_mb=vram_free_mb,
        vram_allocated_mb=vram_allocated_mb,
        vram_reserved_mb=vram_reserved_mb,
        vram_usage_percent=vram_usage_pct,
        cuda_version=cuda_ver,
    )


def calculate_effective_memory_bandwidth(model_size_mb: float, tokens_per_sec: float) -> float:
    """
    Calculate achieved memory bandwidth in GB/s during autoregressive token decode.
    
    Formula:
        Bandwidth (GB/s) = (Model Size in GB) * (Decode Tokens / sec)
    
    In autoregressive generation, memory bandwidth is the primary bottleneck because
    every new token requires streaming all model weights through memory once.
    """
    if tokens_per_sec <= 0 or model_size_mb <= 0:
        return 0.0
    model_size_gb = model_size_mb / 1024.0
    bandwidth_gbs = model_size_gb * tokens_per_sec
    return round(bandwidth_gbs, 2)


def calculate_computer_score(
    tokens_per_sec: float,
    ttft_ms: float,
    model_param_count_m: float,
    memory_bandwidth_gbs: float,
    is_gpu: bool = True
) -> dict:
    """
    Compute a normalized Computer Performance Index Score.
    
    Parameters:
        tokens_per_sec: Generation throughput.
        ttft_ms: Time to first token (prefill latency in ms).
        model_param_count_m: Model size in millions of parameters.
        memory_bandwidth_gbs: Achieved memory bandwidth.
        is_gpu: Whether running on GPU vs CPU.
        
    Returns:
        dict with total score, tier name, and breakdown metrics.
    """
    if tokens_per_sec <= 0:
        return {"score": 0, "tier": "Unrated", "summary": "No generation data"}
    
    # Parameter scaling factor: 500M model = 1.0, 1.5B = 1.8, 7B = 4.5
    param_weight = max(0.5, (model_param_count_m / 500.0) ** 0.6)
    
    # Throughput component (weighted by model size)
    throughput_pts = tokens_per_sec * 8.0 * param_weight
    
    # Bandwidth component
    bandwidth_pts = memory_bandwidth_gbs * 3.5
    
    # Responsiveness component (reward low TTFT, floor at 0)
    ttft_penalty = min(150.0, ttft_ms / 2.0)
    latency_pts = max(0.0, 150.0 - ttft_penalty)
    
    raw_score = throughput_pts + bandwidth_pts + latency_pts
    final_score = int(round(raw_score))
    
    # Tier classifications
    if final_score >= 1200:
        tier = "Ultra Workstation / Datacenter GPU"
        badge = "Tier S"
    elif final_score >= 800:
        tier = "High-Performance Discrete GPU"
        badge = "Tier A"
    elif final_score >= 450:
        tier = "Mid-Range / Laptop GPU"
        badge = "Tier B"
    elif final_score >= 200:
        tier = "Entry-Level GPU / Fast CPU"
        badge = "Tier C"
    else:
        tier = "Standard CPU Tier"
        badge = "Tier D"
        
    return {
        "score": final_score,
        "tier": tier,
        "badge": badge,
        "breakdown": {
            "throughput_points": round(throughput_pts, 1),
            "bandwidth_points": round(bandwidth_pts, 1),
            "latency_points": round(latency_pts, 1),
        }
    }
