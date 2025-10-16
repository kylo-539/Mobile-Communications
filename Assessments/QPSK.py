import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import pandas as pd

# QPSK - Quadrature Phase Shift Keying Implementation
print("QPSK Modulation and Demodulation Simulation")
print("=" * 50)

# Parameters
num_bits = 100000  # Total number of bits
fc = 3            # Carrier frequency (Hz)
A = 5             # Signal amplitude
fs = 100          # Sampling frequency (Hz)
t_symbol = 2      # Symbol duration (2 bits per symbol in QPSK)
t_bit = t_symbol / 2  # Bit duration

# Generate random bit stream
bits = np.random.randint(0, 2, num_bits)
print(f"Generated {num_bits} random bits")

# Convert bits to symbols (2 bits per symbol for QPSK)
# Ensure even number of bits
if len(bits) % 2 != 0:
    bits = bits[:-1]  # Remove last bit if odd number
    num_bits = len(bits)

num_symbols = num_bits // 2  # Each symbol represents 2 bits
symbols = np.zeros(num_symbols, dtype=int)  # Array to store symbol values

# Map bit pairs to symbols: 00->0, 01->1, 10->2, 11->3
for i in range(num_symbols):
    bit_pair = bits[2*i:2*i+2]  # Extract 2 consecutive bits
    symbols[i] = bit_pair[0] * 2 + bit_pair[1]  # Convert binary pair to decimal

print(f"Created {num_symbols} QPSK symbols")
print(f"First 10 symbols: {symbols[:10]}")

# Time vector for one symbol
t = np.arange(0, t_symbol, 1/fs)  # Time array from 0 to symbol duration
samples_per_symbol = len(t)  # Number of samples per symbol period

# QPSK constellation mapping (Gray coding)
# Symbol 0 (00): phase = π/4 (45°)
# Symbol 1 (01): phase = 3π/4 (135°)  
# Symbol 2 (10): phase = -π/4 (315°)
# Symbol 3 (11): phase = -3π/4 (225°)
constellation_phases = {
    0: np.pi/4,      # 00 -> 45°
    1: 3*np.pi/4,    # 01 -> 135°
    2: -np.pi/4,     # 10 -> 315°
    3: -3*np.pi/4    # 11 -> 225°
}

# Create I and Q constellation points
constellation_I = {k: A * np.cos(phase) for k, phase in constellation_phases.items()}  # In-phase components
constellation_Q = {k: A * np.sin(phase) for k, phase in constellation_phases.items()}  # Quadrature components

print("\nQPSK Constellation Points:")
for sym in range(4):
    bits_str = f"{sym//2}{sym%2}" if sym < 2 else f"{(sym-2)//1+1}{(sym-2)%2}"
    if sym == 0: bits_str = "00"
    elif sym == 1: bits_str = "01"
    elif sym == 2: bits_str = "10"
    elif sym == 3: bits_str = "11"
    print(f"Symbol {sym} ({bits_str}): I={constellation_I[sym]:.2f}, Q={constellation_Q[sym]:.2f}, Phase={constellation_phases[sym]*180/np.pi:.0f}°")

# Generate reference carriers (in-phase and quadrature)
I_carrier = np.cos(2 * np.pi * fc * t)  # In-phase carrier (cosine)
Q_carrier = -np.sin(2 * np.pi * fc * t)  # Quadrature carrier (negative sine for conventional QPSK)

# Energy per bit for QPSK
Eb = A**2 * t_symbol / 2  # Energy per bit

# Eb/N0 range for BER simulation
EbN0_dB = np.arange(0, 13, 1)  # 0 to 12 dB
BER = np.zeros(len(EbN0_dB))

# Arrays to store constellation points for plotting
tx_I_constellation = []
tx_Q_constellation = []
rx_I_constellation = []
rx_Q_constellation = []

print(f"\nStarting BER simulation...")
print(f"Energy per bit (Eb): {Eb:.2f}")

# Main BER simulation loop
for idx, EbN0 in enumerate(EbN0_dB):
    # Generate QPSK signal
    qpsk_signal = []  # Array to store complete modulated signal
    tx_I_symbols = []  # Store transmitted I components for constellation
    tx_Q_symbols = []  # Store transmitted Q components for constellation
    
    for symbol in symbols:
        # Get I and Q components for this symbol
        I_amp = constellation_I[symbol]  # In-phase amplitude
        Q_amp = constellation_Q[symbol]  # Quadrature amplitude
        
        # Generate I and Q waveforms
        I_waveform = I_amp * I_carrier  # Modulate I component
        Q_waveform = Q_amp * Q_carrier  # Modulate Q component
        
        # Combined QPSK signal
        symbol_waveform = I_waveform + Q_waveform  # Add I and Q components
        qpsk_signal.extend(symbol_waveform)  # Append to complete signal
        
        # Store constellation points
        tx_I_symbols.append(I_amp)  # Save I component for plotting
        tx_Q_symbols.append(Q_amp)  # Save Q component for plotting
    
    qpsk_signal = np.array(qpsk_signal)  # Convert to numpy array
    
    # Convert Eb/N0 from dB to linear scale
    EbN0_linear = 10**(EbN0 / 10)
    
    # Compute noise variance
    N0 = Eb / EbN0_linear
    noise_variance = N0 * fs / 2  # For complex AWGN
    noise_std = np.sqrt(noise_variance)
    
    # Add AWGN (Additive White Gaussian Noise)
    noise = noise_std * np.random.randn(len(qpsk_signal))  # Generate Gaussian noise
    received_signal = qpsk_signal + noise  # Add noise to transmitted signal
    
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
        
        for sym in range(4):  # Check all 4 QPSK constellation points
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
        # Convert symbol back to bit pair
        bit1 = symbol // 2  # Most significant bit 
        bit2 = symbol % 2   # Least significant bit
        demod_bits[2*i] = bit1      # Store first bit of pair
        demod_bits[2*i+1] = bit2    # Store second bit of pair
    
    # Calculate BER
    bit_errors = np.sum(bits != demod_bits)  # Count mismatched bits
    BER[idx] = bit_errors / num_bits  # Calculate bit error rate
    
    print(f"QPSK - Eb/N0 = {EbN0:2d} dB → BER = {BER[idx]:.6e} → Bit Errors = {bit_errors}")

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
plt.plot(time_plot, plot_tx_signal, 'b-', linewidth=2, label='Transmitted QPSK Signal')
plt.plot(time_plot, plot_rx_signal, 'r-', alpha=0.6, linewidth=1, label='Received Signal (with noise)')
plt.title(f'QPSK Signals - First {plot_symbols} Symbols (Eb/N0 = {EbN0_plot} dB)', fontsize=14, fontweight='bold')
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

for i in range(plot_symbols):
    symbol = symbols[i]
    I_amp = constellation_I[symbol]
    Q_amp = constellation_Q[symbol]
    
    I_waveform = I_amp * I_carrier
    Q_waveform = Q_amp * Q_carrier
    
    plot_I_signal.extend(I_waveform)
    plot_Q_signal.extend(Q_waveform)

plt.plot(time_plot, plot_I_signal, 'b-', linewidth=2, label='I Component')
plt.plot(time_plot, plot_Q_signal, 'r-', linewidth=2, label='Q Component')
plt.title('QPSK I and Q Components', fontsize=14, fontweight='bold')
plt.xlabel('Time (seconds)', fontweight='bold')
plt.ylabel('Amplitude', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Add vertical lines and symbol labels
for i in range(plot_symbols):
    plt.axvline(x=i*t_symbol, color='green', linestyle='--', alpha=0.5)
    if i < plot_symbols:
        symbol = symbols[i]
        bit_pair = f"{symbol//2}{symbol%2}" if symbol < 2 else f"{1 if symbol >= 2 else 0}{symbol%2}"
        if symbol == 0: bit_pair = "00"
        elif symbol == 1: bit_pair = "01"
        elif symbol == 2: bit_pair = "10"
        elif symbol == 3: bit_pair = "11"
        plt.text(i*t_symbol + t_symbol/2, max(plot_I_signal + plot_Q_signal) * 0.8, 
                f'{bit_pair}', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()

# Plot 2: BER vs Eb/N0 comparison
plt.figure(2, figsize=(12, 8))

# Plot simulated BER
plt.semilogy(EbN0_dB, BER, 'o-', linewidth=2, markersize=8, label='Simulated QPSK', color='blue')

# Theoretical QPSK BER: BER = 0.5 * erfc(sqrt(Eb/N0))
EbN0_linear_theory = 10**(EbN0_dB / 10)
BER_theory = 0.5 * erfc(np.sqrt(EbN0_linear_theory))
plt.semilogy(EbN0_dB, BER_theory, '--', linewidth=2, label='Theoretical QPSK', color='red')

plt.xlabel(r'$E_b/N_0$ (dB)', fontsize=14, fontweight='bold')
plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
plt.title('QPSK BER Performance', fontsize=16, fontweight='bold')
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
fig.suptitle('QPSK Constellation Diagrams', fontsize=18, fontweight='bold')

# Colors for different symbols
symbol_colors = ['blue', 'red', 'green', 'orange']
symbol_labels = ['00', '01', '10', '11']
symbol_markers = ['o', 's', '^', 'D']

# Ideal constellation
ax1.set_title('Ideal QPSK Constellation', fontsize=14, fontweight='bold')
for sym in range(4):
    ax1.scatter(constellation_I[sym], constellation_Q[sym], 
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

# Transmitted constellation (sample)
ax2.set_title('Transmitted Constellation (5000 symbols)', fontsize=14, fontweight='bold')
for sym in range(4):
    sym_I = [tx_I_constellation[i] for i in range(len(tx_I_constellation)) if symbols[i] == sym]
    sym_Q = [tx_Q_constellation[i] for i in range(len(tx_Q_constellation)) if symbols[i] == sym]
    ax2.scatter(sym_I, sym_Q, c=symbol_colors[sym], alpha=0.6, s=20, 
               marker=symbol_markers[sym], label=f'{symbol_labels[sym]}', edgecolors='white', linewidth=0.5)
ax2.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='center', bbox_to_anchor=(0.5, 0.5), framealpha=0.9, fontsize=10)
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Received constellation (with noise)
ax3.set_title('Received Constellation (Eb/N0 = 6 dB)', fontsize=14, fontweight='bold')

# Add colored decision regions for each symbol (quadrants)
ax3.fill([0, 10, 10, 0], [0, 0, 10, 10], alpha=0.1, color=symbol_colors[0])  # Quadrant I (00) - blue
ax3.fill([-10, 0, 0, -10], [0, 0, 10, 10], alpha=0.1, color=symbol_colors[1])  # Quadrant II (01) - red  
ax3.fill([-10, 0, 0, -10], [-10, -10, 0, 0], alpha=0.1, color=symbol_colors[3])  # Quadrant III (11) - orange
ax3.fill([0, 10, 10, 0], [-10, -10, 0, 0], alpha=0.1, color=symbol_colors[2])  # Quadrant IV (10) - green

for sym in range(4):
    sym_I = [rx_I_constellation[i] for i in range(len(rx_I_constellation)) if symbols[i] == sym]
    sym_Q = [rx_Q_constellation[i] for i in range(len(rx_Q_constellation)) if symbols[i] == sym]
    ax3.scatter(sym_I, sym_Q, c=symbol_colors[sym], alpha=0.4, s=15, 
               marker=symbol_markers[sym], label=f'{symbol_labels[sym]}')
ax3.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Decision regions
ax4.set_title('QPSK Decision Regions', fontsize=14, fontweight='bold')
# Plot decision boundaries
ax4.axhline(y=0, color='black', linestyle='-', linewidth=2)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=2)

# Add constellation points
for sym in range(4):
    ax4.scatter(constellation_I[sym], constellation_Q[sym], 
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

# Add colored decision regions for each symbol (quadrants)
ax4.fill([0, 6, 6, 0], [0, 0, 6, 6], alpha=0.1, color=symbol_colors[0])  # Quadrant I (00) - blue
ax4.fill([-6, 0, 0, -6], [0, 0, 6, 6], alpha=0.1, color=symbol_colors[1])  # Quadrant II (01) - red  
ax4.fill([-6, 0, 0, -6], [-6, -6, 0, 0], alpha=0.1, color=symbol_colors[3])  # Quadrant III (11) - orange
ax4.fill([0, 6, 6, 0], [-6, -6, 0, 0], alpha=0.1, color=symbol_colors[2])  # Quadrant IV (10) - green

plt.tight_layout()
plt.show()

print(f"\nQPSK Simulation Summary:")
print(f"• Total bits simulated: {num_bits:,}")
print(f"• Total symbols: {num_symbols:,}")
print(f"• Carrier frequency: {fc} Hz")
print(f"• Symbol duration: {t_symbol} s")
print(f"• Energy per bit: {Eb:.2f}")
print(f"• Best BER achieved: {min(BER):.2e} at {EbN0_dB[np.argmin(BER)]} dB")