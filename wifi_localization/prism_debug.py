import serial
import time

# --- CHANGE THIS TO YOUR PORT ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

print(f"📡 Connecting to {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
except Exception as e:
    print(f"❌ Cannot open port: {e}")
    exit()

print("✅ Connected. Printing RAW lines for 10 seconds...\n")
print("=" * 70)

start = time.time()
total_lines = 0
csi_lines = 0
other_lines = 0

while time.time() - start < 10:
    try:
        raw = ser.readline()
        line = raw.decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        total_lines += 1

        if "CSI" in line.upper():
            csi_lines += 1
            # Print first 5 CSI lines in full so we can see the format
            if csi_lines <= 5:
                print(f"[CSI LINE #{csi_lines}]")
                print(repr(line[:300]))   # repr shows hidden chars
                print(f"  Length: {len(line)} chars")
                print(f"  Has '[': {'[' in line}, Has ']': {']' in line}")
                print()
        else:
            other_lines += 1
            if other_lines <= 3:
                print(f"[OTHER] {repr(line[:150])}")

    except Exception as e:
        print(f"[ERROR] {e}")

print("=" * 70)
print(f"\n📊 Summary over 10 seconds:")
print(f"   Total lines    : {total_lines}")
print(f"   CSI lines      : {csi_lines}  (~{csi_lines/10:.1f} packets/sec)")
print(f"   Other lines    : {other_lines}")
print(f"\n⚠️  At {csi_lines/10:.1f} packets/sec, filling 100-packet buffer takes ~{100/(max(csi_lines,1)/10):.0f} seconds")

ser.close()
