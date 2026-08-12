"""
List all video capture devices by name on Windows.
Uses ffmpeg (if available) or WMI to identify camera names.
"""
import subprocess
import sys

# Method 1: Try ffmpeg to list DirectShow devices
print("=" * 60)
print("  DAFTAR KAMERA (via ffmpeg DirectShow)")  
print("=" * 60)
try:
    result = subprocess.run(
        ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
        capture_output=True, text=True, timeout=10
    )
    # ffmpeg prints device list to stderr
    output = result.stderr
    for line in output.split('\n'):
        if 'DirectShow' in line or '"' in line:
            print(" ", line.strip())
except FileNotFoundError:
    print("  ffmpeg tidak ditemukan, mencoba metode lain...")

# Method 2: WMI via PowerShell  
print()
print("=" * 60)
print("  DAFTAR KAMERA (via WMI)")
print("=" * 60)
try:
    ps_script = """
$devices = Get-CimInstance Win32_PnPEntity | Where-Object {
    $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image'
}
foreach ($d in $devices) {
    Write-Output "$($d.Name) | $($d.PNPClass) | $($d.Status)"
}
"""
    result = subprocess.run(
        ['powershell', '-Command', ps_script],
        capture_output=True, text=True, timeout=10
    )
    if result.stdout.strip():
        for line in result.stdout.strip().split('\n'):
            print(f"  {line.strip()}")
    else:
        print("  Tidak ada device ditemukan via WMI")
except Exception as e:
    print(f"  Error: {e}")

# Method 3: Also check for NVIDIA Broadcast specifically
print()
print("=" * 60)
print("  CEK NVIDIA BROADCAST")
print("=" * 60)
try:
    ps_nvidia = """
$devices = Get-CimInstance Win32_PnPEntity | Where-Object {
    $_.Name -like '*NVIDIA*' -or $_.Name -like '*Broadcast*' -or $_.Name -like '*Virtual*'
}
foreach ($d in $devices) {
    Write-Output "$($d.Name) | Class: $($d.PNPClass)"
}
"""
    result = subprocess.run(
        ['powershell', '-Command', ps_nvidia],
        capture_output=True, text=True, timeout=10
    )
    if result.stdout.strip():
        for line in result.stdout.strip().split('\n'):
            print(f"  {line.strip()}")
    else:
        print("  NVIDIA Broadcast tidak ditemukan via WMI")
        print("  (Virtual camera mungkin terdaftar sebagai software device)")
except Exception as e:
    print(f"  Error: {e}")
