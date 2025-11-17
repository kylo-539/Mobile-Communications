import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import pandas as pd

# 8PSK - Eight Phase Shift Keying Implementation
print("8PSK Modulation and Demodulation Simulation")
print("=" * 50)

# Parameters
num_bits = 100000  # Total number of bits
fc = 3            # Carrier frequency (Hz)
A = 5             # Signal amplitude
fs = 100          # Sampling frequency (Hz)
t_symbol = 3      # Symbol duration (3 bits per symbol in 8PSK)
t_bit = t_symbol / 3  # Bit duration

# Generate random bit stream
bits = np.random.randint(0, 2, num_bits)
print(f"Generated {num_bits} random bits")

# Convert bits to symbols (3 bits per symbol for 8PSK)
# Ensure number of bits is divisible by 3
if len(bits) % 3 != 0:
    bits = bits[:-1]  # Remove last bit(s) if not divisible by 3
    num_bits = len(bits)

num_symbols = num_bits // 3  # Each symbol represents 3 bits
symbols = np.zeros(num_symbols, dtype=int)  # Array to store symbol values

# Map bit triplets to symbols: 000->0, 001->1, 010->2, ..., 111->7
for i in range(num_symbols):
    bit_triplet = bits[3*i:3*i+3]  # Extract 3 consecutive bits
    symbols[i] = bit_triplet[0] * 4 + bit_triplet[1] * 2 + bit_triplet[2]  # Convert 3-bit binary to decimal


print(f"Created {num_symbols} 8PSK symbols")
print(f"First 10 symbols: {symbols[:10]}")

# Time vector for one symbol
t = np.arange(0, t_symbol, 1/fs)  # Time array from 0 to symbol duration
samples_per_symbol = len(t)  # Number of samples per symbol period

# 8PSK constellation mapping (Gray coding)
# Symbol 0 (000): phase = π/8 (22.5°)
# Symbol 1 (001): phase = 3π/8 (67.5°)
# Symbol 2 (010): phase = 5π/8 (112.5°)
# Symbol 3 (011): phase = 7π/8 (157.5°)
# Symbol 4 (100): phase = 9π/8 (202.5°)
# Symbol 5 (101): phase = 11π/8 (247.5°)
# Symbol 6 (110): phase = 13π/8 (292.5°)
# Symbol 7 (111): phase = 15π/8 (337.5°)
constellation_phases = {
    0: np.pi/8,      # 000 -> 22.5°
    1: 3*np.pi/8,    # 001 -> 67.5°
    2: 5*np.pi/8,    # 010 -> 112.5°
    3: 7*np.pi/8,    # 011 -> 157.5°
    4: 9*np.pi/8,    # 100 -> 202.5°
    5: 11*np.pi/8,   # 101 -> 247.5°
    6: 13*np.pi/8,   # 110 -> 292.5°
    7: 15*np.pi/8    # 111 -> 337.5°
}

# Create I and Q constellation points
constellation_I = {k: A * np.cos(phase) for k, phase in constellation_phases.items()}  # In-phase components
constellation_Q = {k: A * np.sin(phase) for k, phase in constellation_phases.items()}  # Quadrature components

print("\n8PSK Constellation Points:")
for sym in range(8):
    # Convert symbol number to 3-bit binary string using bit operations
    bits_str = f"{(sym >> 2) & 1}{(sym >> 1) & 1}{sym & 1}"  # Extract individual bits
    print(f"Symbol {sym} ({bits_str}): I={constellation_I[sym]:.2f}, Q={constellation_Q[sym]:.2f}, Phase={constellation_phases[sym]*180/np.pi:.0f}°")

# Generate reference carriers (in-phase and quadrature)
I_carrier = np.cos(2 * np.pi * fc * t)  # In-phase carrier (cosine)
Q_carrier = -np.sin(2 * np.pi * fc * t)  # Quadrature carrier (negative sine)

# Energy per bit for 8PSK
# Symbol energy: Es = A^2 * t_symbol / 2
# For 8PSK: 3 bits per symbol, so Eb = Es / 3
Es = A**2 * t_symbol / 2  # Energy per symbol
Eb = Es / 3  # Energy per bit (3 bits per symbol in 8PSK)

# Eb/N0 range for BER simulation
EbN0_dB = np.arange(0, 13, 1)  # 0 to 12 dB
BER = np.zeros(len(EbN0_dB))

# Arrays to store constellation points for plotting
tx_I_constellation = []  # Transmitted I components for constellation plots
tx_Q_constellation = []  # Transmitted Q components for constellation plots
rx_I_constellation = []  # Received I components for constellation plots
rx_Q_constellation = []  # Received Q components for constellation plots

print(f"\nStarting BER simulation...")
print(f"Energy per bit (Eb): {Eb:.2f}")

# Main BER simulation loop
for idx, EbN0 in enumerate(EbN0_dB):
    # Generate 8PSK signal
    psk8_signal = []  # Array to store complete modulated signal
    tx_I_symbols = []  # Store transmitted I components for constellation
    tx_Q_symbols = []  # Store transmitted Q components for constellation
    
    for symbol in symbols:
        # Get I and Q components for this symbol
        I_amp = constellation_I[symbol]  # In-phase amplitude
        Q_amp = constellation_Q[symbol]  # Quadrature amplitude
        
        # Generate I and Q waveforms
        I_waveform = I_amp * I_carrier  # Modulate I component
        Q_waveform = Q_amp * Q_carrier  # Modulate Q component
        
        # Combined 8PSK signal
        symbol_waveform = I_waveform + Q_waveform  # Add I and Q components
        psk8_signal.extend(symbol_waveform)  # Append to complete signal
        
        # Store constellation points
        tx_I_symbols.append(I_amp)  # Save I component for plotting
        tx_Q_symbols.append(Q_amp)  # Save Q component for plotting
    
    psk8_signal = np.array(psk8_signal)  # Convert to numpy array
    
    # Convert Eb/N0 from dB to linear scale
    EbN0_linear = 10**(EbN0 / 10)
    
    # Compute noise variance
    N0 = Eb / EbN0_linear
    noise_variance = N0 * fs / 2  # For complex AWGN
    noise_std = np.sqrt(noise_variance)
    
    # Add AWGN (Additive White Gaussian Noise)
    noise = noise_std * np.random.randn(len(psk8_signal))  # Generate Gaussian noise
    received_signal = psk8_signal + noise  # Add noise to transmitted signal
    
    # Demodulation using correlation
    demod_symbols = np.zeros(num_symbols, dtype=int)  # Array for demodulated symbols
    rx_I_symbols = []  # Store received I components for constellation
    rx_Q_symbols = []  # Store received Q components for constellation
    
    for i in range(num_symbols):  # Process each symbol
        start = i * samples_per_symbol  # Start index for current symbol
        end = (i + 1) * samples_per_symbol  # End index for current symbol
        
        # Extract received symbol
        rx_segment = received_signal[start:end]  # Get samples for this symbol
        
        # Correlate with I and Q carriers
        I_corr = np.sum(rx_segment * I_carrier) * 2 / fs  # Correlate with cosine carrier
        Q_corr = np.sum(rx_segment * Q_carrier) * 2 / fs  # Correlate with sine carrier
        
        rx_I_symbols.append(I_corr)  # Store received I component
        rx_Q_symbols.append(Q_corr)  # Store received Q component
        
        # Find closest constellation point (minimum distance detection)
        min_distance = float('inf')  # Initialize with infinity
        closest_symbol = 0  # Default symbol
        
        for sym in range(8):  # Check all 8 constellation points
            distance = (I_corr - constellation_I[sym])**2 + (Q_corr - constellation_Q[sym])**2  # Euclidean distance squared
            if distance < min_distance:
                min_distance = distance  # Update minimum distance
                closest_symbol = sym  # Update closest symbol
        
        demod_symbols[i] = closest_symbol  # Store detected symbol
    
    # Store constellation points for middle Eb/N0 (6 dB)
    if EbN0 == 6:
        tx_I_constellation = tx_I_symbols[:5000]  # First 5000 symbols
        tx_Q_constellation = tx_Q_symbols[:5000]
        rx_I_constellation = rx_I_symbols[:5000]
        rx_Q_constellation = rx_Q_symbols[:5000]
    
    # Convert symbols back to bits for BER calculation
    demod_bits = np.zeros(num_bits, dtype=int)  # Array for demodulated bits
    for i in range(num_symbols):
        symbol = demod_symbols[i]  # Get demodulated symbol
        # Convert symbol back to 3-bit triplet using bit operations
        bit1 = (symbol >> 2) & 1  # Most significant bit (extract bit 2)
        bit2 = (symbol >> 1) & 1  # Middle bit (extract bit 1)
        bit3 = symbol & 1         # Least significant bit (extract bit 0)
        demod_bits[3*i] = bit1      # Store first bit of triplet
        demod_bits[3*i+1] = bit2    # Store second bit of triplet
        demod_bits[3*i+2] = bit3    # Store third bit of triplet
    
    # Calculate BER
    bit_errors = np.sum(bits != demod_bits)  # Count mismatched bits
    BER[idx] = bit_errors / num_bits  # Calculate bit error rate
    
    print(f"8PSK - Eb/N0 = {EbN0:2d} dB → BER = {BER[idx]:.6e} → Bit Errors = {bit_errors}")

print("\nBER simulation completed!")

# Plot 1: First 10 symbols of transmitted and received signals
plt.figure(1, figsize=(15, 10))

# Generate signals for plotting (at 6 dB Eb/N0)
plot_symbols = 10  # Number of symbols to plot
EbN0_plot = 6  # SNR for plotting in dB
EbN0_linear = 10**(EbN0_plot / 10)  # Convert to linear scale
N0 = Eb / EbN0_linear  # Noise power spectral density
noise_variance = N0 * fs / 2  # Noise variance for discrete-time
noise_std = np.sqrt(noise_variance)  # Noise standard deviation

# Generate clean and noisy signals for first 10 symbols
plot_tx_signal = []  # Transmitted signal for plotting
plot_rx_signal = []  # Received signal for plotting
time_plot = []  # Time vector for plotting

for i in range(plot_symbols):  # Generate each symbol
    symbol = symbols[i]  # Get symbol value
    I_amp = constellation_I[symbol]  # Get I amplitude
    Q_amp = constellation_Q[symbol]  # Get Q amplitude
    
    I_waveform = I_amp * I_carrier  # Generate I waveform
    Q_waveform = Q_amp * Q_carrier  # Generate Q waveform
    symbol_waveform = I_waveform + Q_waveform  # Combine I and Q
    
    plot_tx_signal.extend(symbol_waveform)  # Add to transmitted signal
    time_plot.extend(t + i * t_symbol)  # Add time points with offset

plot_tx_signal = np.array(plot_tx_signal)  # Convert to numpy array
noise = noise_std * np.random.randn(len(plot_tx_signal))  # Generate noise for plotting
plot_rx_signal = plot_tx_signal + noise  # Add noise to create received signal

# Plot transmitted and received signals
plt.subplot(2, 1, 1)
plt.plot(time_plot, plot_tx_signal, 'b-', linewidth=2, label='Transmitted 8PSK Signal')
plt.plot(time_plot, plot_rx_signal, 'r-', alpha=0.6, linewidth=1, label='Received Signal (with noise)')
plt.title(f'8PSK Signals - First {plot_symbols} Symbols (Eb/N0 = {EbN0_plot} dB)', fontsize=14, fontweight='bold')
plt.xlabel('Time (seconds)', fontweight='bold')
plt.ylabel('Amplitude', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Add vertical lines to show symbol boundaries
for i in range(1, plot_symbols):
    plt.axvline(x=i*t_symbol, color='green', linestyle='--', alpha=0.5)

# Plot I and Q components separately
plt.subplot(2, 1, 2)
plot_I_signal = []
plot_Q_signal = []

for i in range(plot_symbols):  # Generate I and Q components for plotting
    symbol = symbols[i]  # Get symbol value
    I_amp = constellation_I[symbol]  # Get I amplitude
    Q_amp = constellation_Q[symbol]  # Get Q amplitude
    
    I_waveform = I_amp * I_carrier  # Generate I component waveform
    Q_waveform = Q_amp * Q_carrier  # Generate Q component waveform
    
    plot_I_signal.extend(I_waveform)  # Add to I signal array
    plot_Q_signal.extend(Q_waveform)  # Add to Q signal array

plt.plot(time_plot, plot_I_signal, 'b-', linewidth=2, label='I Component')
plt.plot(time_plot, plot_Q_signal, 'r-', linewidth=2, label='Q Component')
plt.title('8PSK I and Q Components', fontsize=14, fontweight='bold')
plt.xlabel('Time (seconds)', fontweight='bold')
plt.ylabel('Amplitude', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Add vertical lines and symbol labels
for i in range(plot_symbols):
    plt.axvline(x=i*t_symbol, color='green', linestyle='--', alpha=0.5)
    if i < plot_symbols:
        symbol = symbols[i]  # Get symbol value
        # Convert symbol to 3-bit string for 8PSK using bit operations
        bit_triplet = f"{(symbol >> 2) & 1}{(symbol >> 1) & 1}{symbol & 1}"  # Extract individual bits
        plt.text(i*t_symbol + t_symbol/2, max(plot_I_signal + plot_Q_signal) * 0.8, 
                f'{bit_triplet}', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()

# Plot 2: BER vs Eb/N0 comparison
plt.figure(2, figsize=(12, 8))

# Plot simulated BER
plt.semilogy(EbN0_dB, BER, 'o-', linewidth=2, markersize=8, label='Simulated 8PSK', color='blue')

# Theoretical 8PSK BER: BER ≈ (2/3) * erfc(sqrt(3*Eb/N0*sin²(π/8)))
EbN0_linear_theory = 10**(EbN0_dB / 10)  # Convert dB to linear scale
sin_pi_8 = np.sin(np.pi/8)  # sin(π/8) for minimum distance calculation
BER_theory = (2/3) * erfc(np.sqrt(3 * EbN0_linear_theory * sin_pi_8**2))  # Theoretical BER formula
plt.semilogy(EbN0_dB, BER_theory, '--', linewidth=2, label='Theoretical 8PSK', color='red')

plt.xlabel(r'$E_b/N_0$ (dB)', fontsize=14, fontweight='bold')
plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
plt.title('8PSK BER Performance', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, which='both', alpha=0.3)
plt.ylim(1e-6, 1)

# Add text box with key information
textstr = f'Simulation Parameters:\n• {num_symbols:,} symbols ({num_bits:,} bits)\n• Carrier freq: {fc} Hz\n• Symbol duration: {t_symbol} s'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Plot 3: Enhanced Constellation Diagrams
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('8PSK Constellation Diagrams', fontsize=18, fontweight='bold')

# Colors for different symbols
symbol_colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
symbol_labels = ['000', '001', '010', '011', '100', '101', '110', '111']
symbol_markers = ['o', 's', '^', 'D', 'P', '*', 'X', '+']

# Ideal constellation
ax1.set_title('Ideal 8PSK Constellation', fontsize=14, fontweight='bold')
for sym in range(8):
    ax1.scatter(constellation_I[sym], constellation_Q[sym], 
               c=symbol_colors[sym], s=200, marker=symbol_markers[sym], 
               label=f'Symbol {sym} ({symbol_labels[sym]})')
ax1.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.5), framealpha=0.9, fontsize=10)
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Transmitted constellation (sample)
ax2.set_title('Transmitted Constellation (5000 symbols)', fontsize=14, fontweight='bold')
for sym in range(8):
    sym_I = [tx_I_constellation[i] for i in range(len(tx_I_constellation)) if symbols[i] == sym]
    sym_Q = [tx_Q_constellation[i] for i in range(len(tx_Q_constellation)) if symbols[i] == sym]
    ax2.scatter(sym_I, sym_Q, c=symbol_colors[sym], alpha=0.6, s=20, 
               marker=symbol_markers[sym], label=f'{symbol_labels[sym]}')
ax2.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Received constellation (with noise)
ax3.set_title('Received Constellation (Eb/N0 = 6 dB)', fontsize=14, fontweight='bold')
for sym in range(8):
    sym_I = [rx_I_constellation[i] for i in range(len(rx_I_constellation)) if symbols[i] == sym]
    sym_Q = [rx_Q_constellation[i] for i in range(len(rx_Q_constellation)) if symbols[i] == sym]
    ax3.scatter(sym_I, sym_Q, c=symbol_colors[sym], alpha=0.4, s=15, 
               marker=symbol_markers[sym], label=f'{symbol_labels[sym]}')

# Add ideal constellation points (centered on received symbol clouds)
for sym in range(8):
    sym_I = [rx_I_constellation[i] for i in range(len(rx_I_constellation)) if symbols[i] == sym]
    sym_Q = [rx_Q_constellation[i] for i in range(len(rx_Q_constellation)) if symbols[i] == sym]
    if len(sym_I) > 0:  # Only if symbols of this type exist
        # Calculate centroid (average position) of received symbols
        mean_I = np.mean(sym_I)
        mean_Q = np.mean(sym_Q)
        ax3.scatter(mean_I, mean_Q, 
                   c=symbol_colors[sym], s=120, marker='s', 
                   edgecolors='black', linewidth=3, zorder=10)

ax3.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Decision regions
ax4.set_title('8PSK Decision Regions', fontsize=14, fontweight='bold')
# Plot decision boundaries - these are at the midpoints between symbols
for i in range(8):
    # Decision boundaries at π/8, 3π/8, 5π/8, 7π/8, 9π/8, 11π/8, 13π/8, 15π/8
    angle = (2*i) * np.pi/8  # Boundary angles between symbols
    x_line = 5 * np.cos(angle)
    y_line = 5 * np.sin(angle)
    ax4.plot([0, x_line], [0, y_line], 'k--', alpha=0.7, linewidth=2)

# Add constellation points at their actual symbol locations
for sym in range(8):
    ax4.scatter(constellation_I[sym], constellation_Q[sym], 
               c=symbol_colors[sym], s=200, marker=symbol_markers[sym], 
               label=f'{symbol_labels[sym]}', edgecolors='black', linewidth=2, zorder=10)

# Add region labels positioned closer to constellation points
label_radius = 2.8  # Closer to constellation points
for sym in range(8):
    angle = (sym + 0.5) * np.pi/4  # Each symbol at its actual angle (π/8, 3π/8, 5π/8, etc.)
    x_pos = label_radius * np.cos(angle)
    y_pos = label_radius * np.sin(angle)
    ax4.text(x_pos, y_pos, symbol_labels[sym], fontsize=12, fontweight='bold', 
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor=symbol_colors[sym], alpha=0.8))

ax4.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-6, 6)
ax4.set_ylim(-6, 6)

# Add colored decision regions as pie-slice wedges with specific boundaries
radius = 6  # Radius for drawing decision regions

# Define exact decision boundaries for each symbol
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
    
    # Create pie slice using wedge
    angles = np.linspace(start_angle, end_angle, 20)
    x_coords = [0] + [radius * np.cos(angle) for angle in angles] + [0]
    y_coords = [0] + [radius * np.sin(angle) for angle in angles] + [0]
    
    ax4.fill(x_coords, y_coords, alpha=0.15, color=symbol_colors[sym])

plt.tight_layout()
plt.show()

print(f"\n8PSK Simulation Summary:")
print(f"• Total bits simulated: {num_bits:,}")
print(f"• Total symbols: {num_symbols:,}")
print(f"• Carrier frequency: {fc} Hz")
print(f"• Symbol duration: {t_symbol} s")
print(f"• Energy per bit: {Eb:.2f}")
print(f"• Best BER achieved: {min(BER):.2e} at {EbN0_dB[np.argmin(BER)]} dB")