import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import pandas as pd

# ================== COMMON PARAMETERS AND BIT STREAM GENERATION ==================
print("Digital Modulation Schemes Comparison")
print("=" * 60)

# Common parameters
num_bits = 100000  # Total number of bits for simulation
fc = 3            # Carrier frequency (Hz)
A = 5             # Signal amplitude
fs = 100          # Sampling frequency (Hz)

# Generate single random bit stream for all modulation schemes
np.random.seed(42)  # Set seed for reproducible results
bits = np.random.randint(0, 2, num_bits)
print(f"Generated {num_bits} random bits for all modulation schemes")
print(f"First 20 bits: {bits[:20]}")

# Common Eb/N0 range (0-15 dB)
EbN0_dB_range = np.arange(0, 16, 1)  # 0 to 15 dB
print(f"Eb/N0 range: {EbN0_dB_range[0]} to {EbN0_dB_range[-1]} dB")

# ================== BASK - BINARY AMPLITUDE SHIFT KEYING ==================
print("\n--- BASK Modulation ---")

# BASK specific parameters
t_bit_bask = 1     # Bit duration for BASK
A1 = 5             # Amplitude for bit = 1
A0 = 0             # Amplitude for bit = 0

# Time vector for one BASK bit
t_bask = np.arange(0, t_bit_bask, 1/fs)
samples_per_bit_bask = len(t_bask)
reference_bask = np.cos(2 * np.pi * fc * t_bask)

# Energy per bit calculation for BASK (Unipolar signaling)
# For unipolar BASK: Eb = (A1^2 * P1 + A0^2 * P0) * T_bit / 2
# Assuming P1 = P0 = 0.5: Eb = (A1^2 * 0.5 + 0 * 0.5) * T_bit / 2 = A1^2 * T_bit / 4
Eb_bask = (A1**2) * t_bit_bask / 4  # Corrected for unipolar BASK
correlation_threshold = A1 * t_bit_bask / 4

# BASK BER simulation
BER_bask = np.zeros(len(EbN0_dB_range))
tx_constellation_bask = []
rx_constellation_bask = []

# BASK Main Simulation Loop
for idx, EbN0 in enumerate(EbN0_dB_range):
    bask_signal = []
    for bit in bits:  # BASK Modulation using common bit stream
        if bit == 1:
            a = A1 * np.cos(2 * np.pi * fc * t_bask)  # High amplitude carrier for bit 1
        else:
            a = A0 * np.cos(2 * np.pi * fc * t_bask)  # Zero amplitude (no carrier) for bit 0
        bask_signal.extend(a)
    bask_signal = np.array(bask_signal)  # Convert to numpy array for processing

    # Convert Eb/N0 from dB to linear scale
    EbN0_linear = 10**(EbN0 / 10)

    # Compute noise variance
    N0 = Eb_bask / EbN0_linear
    noise_variance = N0 * fs / 2
    noise_std = np.sqrt(noise_variance)

    # Add AWGN (Additive White Gaussian Noise)
    noise = noise_std * np.random.randn(len(bask_signal))
    received = bask_signal + noise

    # Demodulation and constellation collection
    demod = np.zeros(len(bits))  # Array to store demodulated bits
    tx_symbols = []  # Store transmitted constellation points
    rx_symbols = []  # Store received constellation points
    
    for i in range(len(bits)):  # Process each bit period
        start = i * samples_per_bit_bask  # Start index for current bit
        end = (i + 1) * samples_per_bit_bask  # End index for current bit
        
        # Transmitted symbol (for constellation)
        tx_segment = bask_signal[start:end]
        tx_symbol = np.sum(tx_segment * reference_bask) / fs
        tx_symbols.append(tx_symbol)
        
        # Received symbol (for constellation) 
        rx_segment = received[start:end]
        rx_symbol = np.sum(rx_segment * reference_bask) / fs
        rx_symbols.append(rx_symbol)
        
        # Decision using correlation threshold
        demod[i] = 1 if rx_symbol > correlation_threshold else 0

    # Store constellation points for the middle Eb/N0 value (7 dB)
    if EbN0 == 7:
        tx_constellation_bask = tx_symbols[:20]  # First 20 symbols
        rx_constellation_bask = rx_symbols[:20]

    # BER Calculation
    bit_errors = np.sum(bits != demod)  # Count mismatched bits
    BER_bask[idx] = bit_errors / num_bits  # Calculate bit error rate
    print(f"BASK - Eb/N0 = {EbN0} dB → BER = {BER_bask[idx]:.6e} → Bit Errors = {bit_errors}")

print("BASK simulation completed!")

# ================== QPSK - QUADRATURE PHASE SHIFT KEYING ==================
print("\n--- QPSK Modulation ---")

# QPSK specific parameters
t_symbol_qpsk = 2      # Symbol duration (2 bits per symbol in QPSK)
t_bit_qpsk = t_symbol_qpsk / 2  # Bit duration

# Convert bits to QPSK symbols (2 bits per symbol)
qpsk_bits = bits[:len(bits)//2*2]  # Ensure even number of bits
num_qpsk_symbols = len(qpsk_bits) // 2 # Number of QPSK symbols
qpsk_symbols = np.zeros(num_qpsk_symbols, dtype=int) # Array to hold QPSK symbols

# Map bit pairs to symbols: 00->0, 01->1, 10->2, 11->3
for i in range(num_qpsk_symbols):
    bit_pair = qpsk_bits[2*i:2*i+2] 
    qpsk_symbols[i] = bit_pair[0] * 2 + bit_pair[1] # Decimal representation

# Time vector for one QPSK symbol
t_qpsk = np.arange(0, t_symbol_qpsk, 1/fs)
samples_per_symbol_qpsk = len(t_qpsk)

# QPSK constellation mapping (Gray coding)
constellation_phases_qpsk = {
    0: np.pi/4,      # 00 -> 45°
    1: 3*np.pi/4,    # 01 -> 135°
    2: -np.pi/4,     # 10 -> 315°
    3: -3*np.pi/4    # 11 -> 225°
}

# Create I and Q constellation points for QPSK
constellation_I_qpsk = {k: A * np.cos(phase) for k, phase in constellation_phases_qpsk.items()}
constellation_Q_qpsk = {k: A * np.sin(phase) for k, phase in constellation_phases_qpsk.items()}

# Generate reference carriers for QPSK
I_carrier_qpsk = np.cos(2 * np.pi * fc * t_qpsk)
Q_carrier_qpsk = -np.sin(2 * np.pi * fc * t_qpsk)

# Energy per bit for QPSK
# For QPSK: Each symbol carries 2 bits, Eb = Es/2 where Es = A^2 * T_symbol / 2
Eb_qpsk = (A**2 * t_symbol_qpsk / 2) / 2  # Corrected: Energy per symbol divided by 2 bits

# QPSK BER simulation
BER_qpsk = np.zeros(len(EbN0_dB_range))
tx_I_constellation_qpsk = []
tx_Q_constellation_qpsk = []
rx_I_constellation_qpsk = []
rx_Q_constellation_qpsk = []

# QPSK Main Simulation Loop
for idx, EbN0 in enumerate(EbN0_dB_range):
    qpsk_signal = []
    tx_I_symbols = []
    tx_Q_symbols = []
    
    for symbol in qpsk_symbols:
        # Get I and Q components for this symbol
        I_amp = constellation_I_qpsk[symbol]
        Q_amp = constellation_Q_qpsk[symbol]
        
        # Generate I and Q waveforms
        I_waveform = I_amp * I_carrier_qpsk
        Q_waveform = Q_amp * Q_carrier_qpsk
        
        # Combined QPSK signal
        symbol_waveform = I_waveform + Q_waveform
        qpsk_signal.extend(symbol_waveform)
        
        # Store constellation points
        tx_I_symbols.append(I_amp)
        tx_Q_symbols.append(Q_amp)
    
    qpsk_signal = np.array(qpsk_signal)
    
    # Convert Eb/N0 from dB to linear scale
    EbN0_linear = 10**(EbN0 / 10)
    
    # Compute noise variance
    N0 = Eb_qpsk / EbN0_linear
    noise_variance = N0 * fs / 2
    noise_std = np.sqrt(noise_variance)
    
    # Add AWGN
    noise = noise_std * np.random.randn(len(qpsk_signal))
    received_signal = qpsk_signal + noise
    
    # Demodulation using correlation
    demod_symbols = np.zeros(num_qpsk_symbols, dtype=int)
    rx_I_symbols = []
    rx_Q_symbols = []
    
    for i in range(num_qpsk_symbols):
        start = i * samples_per_symbol_qpsk
        end = (i + 1) * samples_per_symbol_qpsk
        
        # Extract received symbol
        rx_segment = received_signal[start:end]
        
        # Correlate with I and Q carriers
        I_corr = np.sum(rx_segment * I_carrier_qpsk) * 2 / fs
        Q_corr = np.sum(rx_segment * Q_carrier_qpsk) * 2 / fs
        
        rx_I_symbols.append(I_corr)
        rx_Q_symbols.append(Q_corr)
        
        # Find closest constellation point
        min_distance = float('inf')
        closest_symbol = 0
        
        for sym in range(4):
            distance = (I_corr - constellation_I_qpsk[sym])**2 + (Q_corr - constellation_Q_qpsk[sym])**2
            if distance < min_distance:
                min_distance = distance
                closest_symbol = sym
        
        demod_symbols[i] = closest_symbol
    
    # Store constellation points for middle Eb/N0 (7 dB)
    if EbN0 == 7:
        tx_I_constellation_qpsk = tx_I_symbols[:20]
        tx_Q_constellation_qpsk = tx_Q_symbols[:20]
        rx_I_constellation_qpsk = rx_I_symbols[:20]
        rx_Q_constellation_qpsk = rx_Q_symbols[:20]
    
    # Convert symbols back to bits for BER calculation
    demod_bits = np.zeros(len(qpsk_bits), dtype=int)
    for i in range(num_qpsk_symbols):
        symbol = demod_symbols[i]
        bit1 = symbol // 2
        bit2 = symbol % 2
        demod_bits[2*i] = bit1
        demod_bits[2*i+1] = bit2
    
    # Calculate BER
    bit_errors = np.sum(qpsk_bits != demod_bits)
    BER_qpsk[idx] = bit_errors / len(qpsk_bits)
    
    print(f"QPSK - Eb/N0 = {EbN0:2d} dB → BER = {BER_qpsk[idx]:.6e} → Bit Errors = {bit_errors}")

print("QPSK simulation completed!")

# ================== 8PSK - EIGHT PHASE SHIFT KEYING ==================
print("\n--- 8PSK Modulation ---")

# 8PSK specific parameters
t_symbol_8psk = 3      # Symbol duration (3 bits per symbol in 8PSK)
t_bit_8psk = t_symbol_8psk / 3  # Bit duration

# Convert bits to 8PSK symbols (3 bits per symbol)
psk8_bits = bits[:len(bits)//3*3]  # Ensure number of bits is divisible by 3
num_8psk_symbols = len(psk8_bits) // 3
psk8_symbols = np.zeros(num_8psk_symbols, dtype=int)

# Map bit triplets to symbols: 000->0, 001->1, 010->2, ..., 111->7
for i in range(num_8psk_symbols):
    bit_triplet = psk8_bits[3*i:3*i+3]
    psk8_symbols[i] = bit_triplet[0] * 4 + bit_triplet[1] * 2 + bit_triplet[2]

# Time vector for one 8PSK symbol
t_8psk = np.arange(0, t_symbol_8psk, 1/fs)
samples_per_symbol_8psk = len(t_8psk)

# 8PSK constellation mapping
constellation_phases_8psk = {
    0: np.pi/8,      # 000 -> 22.5°
    1: 3*np.pi/8,    # 001 -> 67.5°
    2: 5*np.pi/8,    # 010 -> 112.5°
    3: 7*np.pi/8,    # 011 -> 157.5°
    4: 9*np.pi/8,    # 100 -> 202.5°
    5: 11*np.pi/8,   # 101 -> 247.5°
    6: 13*np.pi/8,   # 110 -> 292.5°
    7: 15*np.pi/8    # 111 -> 337.5°
}

# Create I and Q constellation points for 8PSK
constellation_I_8psk = {k: A * np.cos(phase) for k, phase in constellation_phases_8psk.items()}
constellation_Q_8psk = {k: A * np.sin(phase) for k, phase in constellation_phases_8psk.items()}

# Generate reference carriers for 8PSK
I_carrier_8psk = np.cos(2 * np.pi * fc * t_8psk)
Q_carrier_8psk = -np.sin(2 * np.pi * fc * t_8psk)

# Energy per bit for 8PSK
Es_8psk = A**2 * t_symbol_8psk / 2  # Energy per symbol
Eb_8psk = Es_8psk / 3  # Energy per bit (3 bits per symbol)

# 8PSK BER simulation
BER_8psk = np.zeros(len(EbN0_dB_range))
tx_I_constellation_8psk = []
tx_Q_constellation_8psk = []
rx_I_constellation_8psk = []
rx_Q_constellation_8psk = []

# 8PSK Main Simulation Loop
for idx, EbN0 in enumerate(EbN0_dB_range):
    psk8_signal = []
    tx_I_symbols = []
    tx_Q_symbols = []
    
    for symbol in psk8_symbols:
        # Get I and Q components for this symbol
        I_amp = constellation_I_8psk[symbol]
        Q_amp = constellation_Q_8psk[symbol]
        
        # Generate I and Q waveforms
        I_waveform = I_amp * I_carrier_8psk
        Q_waveform = Q_amp * Q_carrier_8psk
        
        # Combined 8PSK signal
        symbol_waveform = I_waveform + Q_waveform
        psk8_signal.extend(symbol_waveform)
        
        # Store constellation points
        tx_I_symbols.append(I_amp)
        tx_Q_symbols.append(Q_amp)
    
    psk8_signal = np.array(psk8_signal)
    
    # Convert Eb/N0 from dB to linear scale
    EbN0_linear = 10**(EbN0 / 10)
    
    # Compute noise variance
    N0 = Eb_8psk / EbN0_linear
    noise_variance = N0 * fs / 2
    noise_std = np.sqrt(noise_variance)
    
    # Add AWGN
    noise = noise_std * np.random.randn(len(psk8_signal))
    received_signal = psk8_signal + noise
    
    # Demodulation using correlation
    demod_symbols = np.zeros(num_8psk_symbols, dtype=int)
    rx_I_symbols = []
    rx_Q_symbols = []
    
    for i in range(num_8psk_symbols):
        start = i * samples_per_symbol_8psk
        end = (i + 1) * samples_per_symbol_8psk
        
        # Extract received symbol
        rx_segment = received_signal[start:end]
        
        # Correlate with I and Q carriers
        I_corr = np.sum(rx_segment * I_carrier_8psk) * 2 / fs
        Q_corr = np.sum(rx_segment * Q_carrier_8psk) * 2 / fs
        
        rx_I_symbols.append(I_corr)
        rx_Q_symbols.append(Q_corr)
        
        # Find closest constellation point
        min_distance = float('inf')
        closest_symbol = 0
        
        for sym in range(8):
            distance = (I_corr - constellation_I_8psk[sym])**2 + (Q_corr - constellation_Q_8psk[sym])**2
            if distance < min_distance:
                min_distance = distance
                closest_symbol = sym
        
        demod_symbols[i] = closest_symbol
    
    # Store constellation points for middle Eb/N0 (7 dB)
    if EbN0 == 7:
        tx_I_constellation_8psk = tx_I_symbols[:20]
        tx_Q_constellation_8psk = tx_Q_symbols[:20]
        rx_I_constellation_8psk = rx_I_symbols[:20]
        rx_Q_constellation_8psk = rx_Q_symbols[:20]
    
    # Convert symbols back to bits for BER calculation
    demod_bits = np.zeros(len(psk8_bits), dtype=int)
    for i in range(num_8psk_symbols):
        symbol = demod_symbols[i]
        bit1 = (symbol >> 2) & 1
        bit2 = (symbol >> 1) & 1
        bit3 = symbol & 1
        demod_bits[3*i] = bit1
        demod_bits[3*i+1] = bit2
        demod_bits[3*i+2] = bit3
    
    # Calculate BER
    bit_errors = np.sum(psk8_bits != demod_bits)
    BER_8psk[idx] = bit_errors / len(psk8_bits)
    
    print(f"8PSK - Eb/N0 = {EbN0:2d} dB → BER = {BER_8psk[idx]:.6e} → Bit Errors = {bit_errors}")

print("8PSK simulation completed!")

# ================== BER COMPARISON AND DEBUGGING ==================
print("\n--- BER Values Comparison ---")
print("Eb/N0 (dB) | BASK Sim    | BASK Theory | QPSK Sim    | QPSK Theory | 8PSK Sim    | 8PSK Theory")
print("-" * 95)

# Calculate theoretical values for comparison
EbN0_linear_debug = 10**(EbN0_dB_range / 10)
BER_theory_bask_debug = 0.5 * erfc(np.sqrt(EbN0_linear_debug / 2))
BER_theory_qpsk_debug = 0.5 * erfc(np.sqrt(EbN0_linear_debug))
sin_pi_8_debug = np.sin(np.pi/8)
BER_theory_8psk_debug = (2/3) * erfc(np.sqrt(3 * EbN0_linear_debug * sin_pi_8_debug**2))

for i, EbN0 in enumerate(EbN0_dB_range[::2]):  # Print every other value to avoid clutter
    idx = i * 2  # Adjust index for every other value
    if idx < len(BER_bask):
        print(f"{EbN0:8.0f}   | {BER_bask[idx]:10.3e} | {BER_theory_bask_debug[idx]:10.3e} | "
              f"{BER_qpsk[idx]:10.3e} | {BER_theory_qpsk_debug[idx]:10.3e} | "
              f"{BER_8psk[idx]:10.3e} | {BER_theory_8psk_debug[idx]:10.3e}")

print(f"\nEnergy per bit values:")
print(f"• BASK Eb: {Eb_bask:.4f}")
print(f"• QPSK Eb: {Eb_qpsk:.4f}")
print(f"• 8PSK Eb: {Eb_8psk:.4f}")

# ================== PLOTTING SECTION ==================
print("\n--- Generating Plots ---")

# Plot 1: Individual BASK Constellation Diagrams
plt.figure(1, figsize=(16, 8))
plt.suptitle("BASK Constellation Diagrams", fontsize=16, fontweight='bold')

# BASK Transmitted constellation
plt.subplot(1, 2, 1)
tx_0_bits = [tx_constellation_bask[i] for i in range(len(tx_constellation_bask)) if bits[i] == 0]
tx_1_bits = [tx_constellation_bask[i] for i in range(len(tx_constellation_bask)) if bits[i] == 1]

# Add vertical jitter for visibility
np.random.seed(42)
jitter_0 = np.random.normal(0, 0.05, len(tx_0_bits))
jitter_1 = np.random.normal(0, 0.05, len(tx_1_bits))

plt.scatter(tx_0_bits, jitter_0, c='darkblue', alpha=0.7, label='Bit 0', s=25, marker='o')
plt.scatter(tx_1_bits, jitter_1, c='darkred', alpha=0.7, label='Bit 1', s=25, marker='^')

plt.xlabel('In-phase Component', fontsize=12, fontweight='bold')
plt.ylabel('Quadrature Component', fontsize=12, fontweight='bold')
plt.title('BASK Transmitted Constellation (20 symbols)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(-0.3, 0.3)
plt.axhline(y=0, color='black', linewidth=1, alpha=0.5)
plt.axvline(x=0, color='black', linewidth=1, alpha=0.5)

# BASK Received constellation
plt.subplot(1, 2, 2)
rx_0_bits = [rx_constellation_bask[i] for i in range(len(rx_constellation_bask)) if bits[i] == 0]
rx_1_bits = [rx_constellation_bask[i] for i in range(len(rx_constellation_bask)) if bits[i] == 1]

jitter_rx_0 = np.random.normal(0, 0.05, len(rx_0_bits))
jitter_rx_1 = np.random.normal(0, 0.05, len(rx_1_bits))

plt.scatter(rx_0_bits, jitter_rx_0, c='lightblue', alpha=0.5, label='Received Bit 0', s=15, marker='o')
plt.scatter(rx_1_bits, jitter_rx_1, c='lightcoral', alpha=0.5, label='Received Bit 1', s=15, marker='^')

# Decision threshold
plt.axvline(x=correlation_threshold, color='green', linestyle='--', linewidth=3, 
           label=f'Decision Threshold = {correlation_threshold:.2f}', zorder=4)

# Add decision regions
plt.axvspan(-10, correlation_threshold, alpha=0.1, color='blue')
plt.axvspan(correlation_threshold, 10, alpha=0.1, color='red')

plt.xlabel('In-phase Component', fontsize=12, fontweight='bold')
plt.ylabel('Quadrature Component', fontsize=12, fontweight='bold')
plt.title('BASK Received Constellation (Eb/N0 = 7 dB, 20 symbols)', fontsize=14, fontweight='bold')
plt.legend(fontsize=9, loc='upper right')
plt.grid(True, alpha=0.3)
plt.ylim(-0.3, 0.3)
plt.axhline(y=0, color='black', linewidth=1, alpha=0.5)
plt.axvline(x=0, color='black', linewidth=1, alpha=0.5)

plt.tight_layout()

# Plot 2: Individual QPSK Constellation Diagrams
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('QPSK Constellation Diagrams', fontsize=18, fontweight='bold')

# Colors and labels for QPSK symbols
symbol_colors = ['blue', 'red', 'green', 'orange']
symbol_labels = ['00', '01', '10', '11']
symbol_markers = ['o', 's', '^', 'D']

# QPSK Ideal constellation
ax1.set_title('Ideal QPSK Constellation', fontsize=14, fontweight='bold')
for sym in range(4):
    ax1.scatter(constellation_I_qpsk[sym], constellation_Q_qpsk[sym], 
               c=symbol_colors[sym], s=200, marker=symbol_markers[sym], 
               label=f'Symbol {sym} ({symbol_labels[sym]})', edgecolors='black', linewidth=2)
ax1.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.5), framealpha=0.9, fontsize=10)
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# QPSK Transmitted constellation
ax2.set_title('QPSK Transmitted Constellation (20 symbols)', fontsize=14, fontweight='bold')
for sym in range(4):
    sym_I = [tx_I_constellation_qpsk[i] for i in range(len(tx_I_constellation_qpsk)) if qpsk_symbols[i] == sym]
    sym_Q = [tx_Q_constellation_qpsk[i] for i in range(len(tx_Q_constellation_qpsk)) if qpsk_symbols[i] == sym]
    ax2.scatter(sym_I, sym_Q, c=symbol_colors[sym], alpha=0.6, s=20, 
               marker=symbol_markers[sym], label=f'{symbol_labels[sym]}', edgecolors='white', linewidth=0.5)
ax2.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='center', bbox_to_anchor=(0.5, 0.5), framealpha=0.9, fontsize=10)
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# QPSK Received constellation
ax3.set_title('QPSK Received Constellation (Eb/N0 = 7 dB)', fontsize=14, fontweight='bold')

# Add colored decision regions for QPSK quadrants
ax3.fill([0, 15, 15, 0], [0, 0, 15, 15], alpha=0.1, color=symbol_colors[0])  # Quadrant I (00)
ax3.fill([-15, 0, 0, -15], [0, 0, 15, 15], alpha=0.1, color=symbol_colors[1])  # Quadrant II (01)
ax3.fill([-15, 0, 0, -15], [-15, -15, 0, 0], alpha=0.1, color=symbol_colors[3])  # Quadrant III (11)
ax3.fill([0, 15, 15, 0], [-15, -15, 0, 0], alpha=0.1, color=symbol_colors[2])  # Quadrant IV (10)

for sym in range(4):
    sym_I = [rx_I_constellation_qpsk[i] for i in range(len(rx_I_constellation_qpsk)) if qpsk_symbols[i] == sym]
    sym_Q = [rx_Q_constellation_qpsk[i] for i in range(len(rx_Q_constellation_qpsk)) if qpsk_symbols[i] == sym]
    ax3.scatter(sym_I, sym_Q, c=symbol_colors[sym], alpha=0.4, s=15, 
               marker=symbol_markers[sym], label=f'{symbol_labels[sym]}')
ax3.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# QPSK Decision regions
ax4.set_title('QPSK Decision Regions', fontsize=14, fontweight='bold')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=2)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=2)

# Add constellation points
for sym in range(4):
    ax4.scatter(constellation_I_qpsk[sym], constellation_Q_qpsk[sym], 
               c=symbol_colors[sym], s=200, marker=symbol_markers[sym], 
               label=f'{symbol_labels[sym]}', edgecolors='black', linewidth=2)

# Add region labels
ax4.text(2.5, 2.5, '00', fontsize=16, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
ax4.text(-2.5, 2.5, '01', fontsize=16, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
ax4.text(-2.5, -2.5, '11', fontsize=16, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
ax4.text(2.5, -2.5, '10', fontsize=16, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

ax4.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-6, 6)
ax4.set_ylim(-6, 6)

# Add colored decision regions
ax4.fill([0, 6, 6, 0], [0, 0, 6, 6], alpha=0.1, color=symbol_colors[0])  # Quadrant I (00)
ax4.fill([-6, 0, 0, -6], [0, 0, 6, 6], alpha=0.1, color=symbol_colors[1])  # Quadrant II (01)
ax4.fill([-6, 0, 0, -6], [-6, -6, 0, 0], alpha=0.1, color=symbol_colors[3])  # Quadrant III (11)
ax4.fill([0, 6, 6, 0], [-6, -6, 0, 0], alpha=0.1, color=symbol_colors[2])  # Quadrant IV (10)

plt.tight_layout()

# Plot 3: Individual 8PSK Constellation Diagrams
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('8PSK Constellation Diagrams', fontsize=18, fontweight='bold')

# Colors and labels for 8PSK symbols
symbol_colors_8psk = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
symbol_labels_8psk = ['000', '001', '010', '011', '100', '101', '110', '111']
symbol_markers_8psk = ['o', 's', '^', 'D', 'P', '*', 'X', '+']

# 8PSK Ideal constellation
ax1.set_title('Ideal 8PSK Constellation', fontsize=14, fontweight='bold')
for sym in range(8):
    ax1.scatter(constellation_I_8psk[sym], constellation_Q_8psk[sym], 
               c=symbol_colors_8psk[sym], s=200, marker=symbol_markers_8psk[sym], 
               label=f'Symbol {sym} ({symbol_labels_8psk[sym]})')
ax1.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.5), framealpha=0.9, fontsize=10)
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# 8PSK Transmitted constellation
ax2.set_title('8PSK Transmitted Constellation (20 symbols)', fontsize=14, fontweight='bold')
for sym in range(8):
    sym_I = [tx_I_constellation_8psk[i] for i in range(len(tx_I_constellation_8psk)) if psk8_symbols[i] == sym]
    sym_Q = [tx_Q_constellation_8psk[i] for i in range(len(tx_Q_constellation_8psk)) if psk8_symbols[i] == sym]
    ax2.scatter(sym_I, sym_Q, c=symbol_colors_8psk[sym], alpha=0.6, s=20, 
               marker=symbol_markers_8psk[sym], label=f'{symbol_labels_8psk[sym]}')
ax2.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# 8PSK Received constellation
ax3.set_title('8PSK Received Constellation (Eb/N0 = 7 dB)', fontsize=14, fontweight='bold')
for sym in range(8):
    sym_I = [rx_I_constellation_8psk[i] for i in range(len(rx_I_constellation_8psk)) if psk8_symbols[i] == sym]
    sym_Q = [rx_Q_constellation_8psk[i] for i in range(len(rx_Q_constellation_8psk)) if psk8_symbols[i] == sym]
    ax3.scatter(sym_I, sym_Q, c=symbol_colors_8psk[sym], alpha=0.4, s=15, 
               marker=symbol_markers_8psk[sym], label=f'{symbol_labels_8psk[sym]}')

# Add ideal constellation points (centered on received symbol clouds)
for sym in range(8):
    sym_I = [rx_I_constellation_8psk[i] for i in range(len(rx_I_constellation_8psk)) if psk8_symbols[i] == sym]
    sym_Q = [rx_Q_constellation_8psk[i] for i in range(len(rx_Q_constellation_8psk)) if psk8_symbols[i] == sym]
    if len(sym_I) > 0:  # Only if symbols of this type exist
        mean_I = np.mean(sym_I)
        mean_Q = np.mean(sym_Q)
        ax3.scatter(mean_I, mean_Q, 
                   c=symbol_colors_8psk[sym], s=120, marker='s', 
                   edgecolors='black', linewidth=3, zorder=10)

ax3.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# 8PSK Decision regions
ax4.set_title('8PSK Decision Regions', fontsize=14, fontweight='bold')

# Plot decision boundaries
for i in range(8):
    angle = (2*i) * np.pi/8  # Boundary angles between symbols
    x_line = 5 * np.cos(angle)
    y_line = 5 * np.sin(angle)
    ax4.plot([0, x_line], [0, y_line], 'k--', alpha=0.7, linewidth=2)

# Add constellation points
for sym in range(8):
    ax4.scatter(constellation_I_8psk[sym], constellation_Q_8psk[sym], 
               c=symbol_colors_8psk[sym], s=200, marker=symbol_markers_8psk[sym], 
               label=f'{symbol_labels_8psk[sym]}', edgecolors='black', linewidth=2, zorder=10)

# Add region labels
label_radius = 2.8
for sym in range(8):
    angle = (sym + 0.5) * np.pi/4
    x_pos = label_radius * np.cos(angle)
    y_pos = label_radius * np.sin(angle)
    ax4.text(x_pos, y_pos, symbol_labels_8psk[sym], fontsize=12, fontweight='bold', 
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor=symbol_colors_8psk[sym], alpha=0.8))

ax4.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-6, 6)
ax4.set_ylim(-6, 6)

# Add colored decision regions as pie-slice wedges
radius = 6
decision_boundaries = {
    0: (0, np.pi/4),              # 000: 0 to π/4
    1: (np.pi/4, np.pi/2),        # 001: π/4 to π/2  
    2: (np.pi/2, 3*np.pi/4),      # 010: π/2 to 3π/4
    3: (3*np.pi/4, np.pi),        # 011: 3π/4 to π
    4: (np.pi, 5*np.pi/4),        # 100: π to 5π/4
    5: (5*np.pi/4, 3*np.pi/2),    # 101: 5π/4 to 3π/2
    6: (3*np.pi/2, 7*np.pi/4),    # 110: 3π/2 to 7π/4
    7: (7*np.pi/4, 2*np.pi)       # 111: 7π/4 to 2π
}

for sym in range(8):
    start_angle, end_angle = decision_boundaries[sym]
    angles = np.linspace(start_angle, end_angle, 20)
    x_coords = [0] + [radius * np.cos(angle) for angle in angles] + [0]
    y_coords = [0] + [radius * np.sin(angle) for angle in angles] + [0]
    ax4.fill(x_coords, y_coords, alpha=0.15, color=symbol_colors_8psk[sym])

plt.tight_layout()

# Plot 4: Combined TX Constellation Diagrams
plt.figure(4, figsize=(18, 6))
plt.suptitle('Transmitted Constellation Diagrams Comparison', fontsize=18, fontweight='bold')

# BASK TX Constellation
plt.subplot(1, 3, 1)
tx_0_bits = [tx_constellation_bask[i] for i in range(len(tx_constellation_bask)) if bits[i] == 0]
tx_1_bits = [tx_constellation_bask[i] for i in range(len(tx_constellation_bask)) if bits[i] == 1]
jitter_0 = np.random.normal(0, 0.05, len(tx_0_bits))
jitter_1 = np.random.normal(0, 0.05, len(tx_1_bits))
plt.scatter(tx_0_bits, jitter_0, c='darkblue', alpha=0.7, label='Bit 0', s=25, marker='o')
plt.scatter(tx_1_bits, jitter_1, c='darkred', alpha=0.7, label='Bit 1', s=25, marker='^')
plt.xlabel('In-phase Component')
plt.ylabel('Quadrature Component')
plt.title('BASK Transmitted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.3, 0.3)

# QPSK TX Constellation
plt.subplot(1, 3, 2)
for sym in range(4):
    sym_I = [tx_I_constellation_qpsk[i] for i in range(len(tx_I_constellation_qpsk)) if qpsk_symbols[i] == sym]
    sym_Q = [tx_Q_constellation_qpsk[i] for i in range(len(tx_Q_constellation_qpsk)) if qpsk_symbols[i] == sym]
    plt.scatter(sym_I, sym_Q, c=symbol_colors[sym], alpha=0.6, s=20, 
               marker=symbol_markers[sym], label=f'{symbol_labels[sym]}')
plt.xlabel('In-phase Component')
plt.ylabel('Quadrature Component')
plt.title('QPSK Transmitted')
plt.legend()
plt.grid(True, alpha=0.3)

# 8PSK TX Constellation
plt.subplot(1, 3, 3)
for sym in range(8):
    sym_I = [tx_I_constellation_8psk[i] for i in range(len(tx_I_constellation_8psk)) if psk8_symbols[i] == sym]
    sym_Q = [tx_Q_constellation_8psk[i] for i in range(len(tx_Q_constellation_8psk)) if psk8_symbols[i] == sym]
    plt.scatter(sym_I, sym_Q, c=symbol_colors_8psk[sym], alpha=0.6, s=20, 
               marker=symbol_markers_8psk[sym], label=f'{symbol_labels_8psk[sym]}')
plt.xlabel('In-phase Component')
plt.ylabel('Quadrature Component')
plt.title('8PSK Transmitted')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Plot 5: Combined RX Constellation Diagrams
plt.figure(5, figsize=(18, 6))
plt.suptitle('Received Constellation Diagrams Comparison (Eb/N0 = 7 dB)', fontsize=18, fontweight='bold')

# BASK RX Constellation
plt.subplot(1, 3, 1)
rx_0_bits = [rx_constellation_bask[i] for i in range(len(rx_constellation_bask)) if bits[i] == 0]
rx_1_bits = [rx_constellation_bask[i] for i in range(len(rx_constellation_bask)) if bits[i] == 1]
jitter_rx_0 = np.random.normal(0, 0.05, len(rx_0_bits))
jitter_rx_1 = np.random.normal(0, 0.05, len(rx_1_bits))
plt.scatter(rx_0_bits, jitter_rx_0, c='lightblue', alpha=0.5, label='Received Bit 0', s=15, marker='o')
plt.scatter(rx_1_bits, jitter_rx_1, c='lightcoral', alpha=0.5, label='Received Bit 1', s=15, marker='^')
plt.axvline(x=correlation_threshold, color='green', linestyle='--', linewidth=3, label='Decision Threshold')
plt.xlabel('In-phase Component')
plt.ylabel('Quadrature Component')
plt.title('BASK Received')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.3, 0.3)

# QPSK RX Constellation
plt.subplot(1, 3, 2)
for sym in range(4):
    sym_I = [rx_I_constellation_qpsk[i] for i in range(len(rx_I_constellation_qpsk)) if qpsk_symbols[i] == sym]
    sym_Q = [rx_Q_constellation_qpsk[i] for i in range(len(rx_Q_constellation_qpsk)) if qpsk_symbols[i] == sym]
    plt.scatter(sym_I, sym_Q, c=symbol_colors[sym], alpha=0.4, s=15, 
               marker=symbol_markers[sym], label=f'{symbol_labels[sym]}')
plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
plt.axvline(x=0, color='black', linestyle='-', linewidth=2)
plt.xlabel('In-phase Component')
plt.ylabel('Quadrature Component')
plt.title('QPSK Received')
plt.legend()
plt.grid(True, alpha=0.3)

# 8PSK RX Constellation
plt.subplot(1, 3, 3)
for sym in range(8):
    sym_I = [rx_I_constellation_8psk[i] for i in range(len(rx_I_constellation_8psk)) if psk8_symbols[i] == sym]
    sym_Q = [rx_Q_constellation_8psk[i] for i in range(len(rx_Q_constellation_8psk)) if psk8_symbols[i] == sym]
    plt.scatter(sym_I, sym_Q, c=symbol_colors_8psk[sym], alpha=0.4, s=15, 
               marker=symbol_markers_8psk[sym], label=f'{symbol_labels_8psk[sym]}')
# Plot decision boundaries
for i in range(8):
    angle = (2*i) * np.pi/8
    x_line = 20 * np.cos(angle)
    y_line = 20 * np.sin(angle)
    plt.plot([0, x_line], [0, y_line], 'k--', alpha=0.7, linewidth=1)
plt.xlabel('In-phase Component')
plt.ylabel('Quadrature Component')
plt.title('8PSK Received')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Plot 6: Combined BER vs Eb/N0 Performance
plt.figure(6, figsize=(12, 8))

# Plot simulated BER curves with distinct markers and colors
plt.semilogy(EbN0_dB_range, BER_bask, 'o-', linewidth=2.5, markersize=10, label='BASK Simulated', 
            color='darkblue', markerfacecolor='lightblue', markeredgecolor='darkblue', markeredgewidth=2)
plt.semilogy(EbN0_dB_range, BER_qpsk, 's-', linewidth=2.5, markersize=10, label='QPSK Simulated', 
            color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markeredgewidth=2)
plt.semilogy(EbN0_dB_range, BER_8psk, '^-', linewidth=2.5, markersize=10, label='8PSK Simulated', 
            color='darkgreen', markerfacecolor='lightgreen', markeredgecolor='darkgreen', markeredgewidth=2)

# Plot theoretical BER curves
EbN0_linear_theory = 10**(EbN0_dB_range / 10)

# Theoretical BASK BER (Unipolar): BER = 0.5 * erfc(sqrt(Eb/2N0))
# For unipolar BASK, the performance is 3dB worse than bipolar
BER_theory_bask = 0.5 * erfc(np.sqrt(EbN0_linear_theory / 2))
plt.semilogy(EbN0_dB_range, BER_theory_bask, '--', linewidth=3, label='BASK Theoretical', color='blue', alpha=0.8)

# Theoretical QPSK BER: BER = 0.5 * erfc(sqrt(Eb/N0))
# QPSK has the same BER as BPSK
BER_theory_qpsk = 0.5 * erfc(np.sqrt(EbN0_linear_theory))
plt.semilogy(EbN0_dB_range, BER_theory_qpsk, '--', linewidth=3, label='QPSK Theoretical', color='red', alpha=0.8)

# Theoretical 8PSK BER: BER ≈ (2/3) * erfc(sqrt(3*Eb/N0*sin²(π/8)))
sin_pi_8 = np.sin(np.pi/8)
BER_theory_8psk = (2/3) * erfc(np.sqrt(3 * EbN0_linear_theory * sin_pi_8**2))
plt.semilogy(EbN0_dB_range, BER_theory_8psk, '--', linewidth=3, label='8PSK Theoretical', color='green', alpha=0.8)

plt.xlabel(r'$E_b/N_0$ (dB)', fontsize=14, fontweight='bold')
plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
plt.title('BER Performance Comparison: BASK vs QPSK vs 8PSK', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, which='both', alpha=0.3)
plt.xlim(0, 15)
plt.ylim(1e-6, 1)

# Add text box with simulation parameters
textstr = f'Simulation Parameters:\n• Bits simulated: {num_bits:,}\n• Carrier freq: {fc} Hz\n• Sampling freq: {fs} Hz'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()

# Display all plots
plt.show()

print("\n" + "="*60)
print("SIMULATION SUMMARY")
print("="*60)
print(f"• Total bits simulated: {num_bits:,}")
print(f"• BASK symbols: {num_bits:,} (1 bit per symbol)")
print(f"• QPSK symbols: {num_qpsk_symbols:,} (2 bits per symbol)")
print(f"• 8PSK symbols: {num_8psk_symbols:,} (3 bits per symbol)")
print(f"• Carrier frequency: {fc} Hz")
print(f"• Sampling frequency: {fs} Hz")
print(f"• Eb/N0 range: {EbN0_dB_range[0]} to {EbN0_dB_range[-1]} dB")
print("\nBest BER Performance (at 15 dB Eb/N0):")
print(f"• BASK: {BER_bask[-1]:.2e}")
print(f"• QPSK: {BER_qpsk[-1]:.2e}")
print(f"• 8PSK: {BER_8psk[-1]:.2e}")