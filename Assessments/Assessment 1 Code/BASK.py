import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert
import pandas as pd

# BASK - Modulation Scheme
num_bits = 100000
sig = np.random.randint(0, 2, num_bits) # Generate a Random stream of 10^6 bits
sig_20 = sig[:20] # First 20 bits of the signal

t_bit = 1  # Bit duration
fc = 3     # Carrier Frequency
A1 = 5     # Amplitude for bit = 1
A0 = 0     # Amplitude for bit = 0
fs = 100   # Sampling Frequency

t = np.arange(0, t_bit, 1/fs)  # Time vector for one bit

samples_per_bit = len(t)  # Number of samples per bit period
reference = np.cos(2 * np.pi * fc * t)  # Reference carrier for correlation receiver (same as transmitter)  

# Energy per bit calculation for BASK
# For unipolar BASK (A0=0, A1=5): Eb = A1^2 * t_bit / 2
Eb = (A1**2) * t_bit / 2

# For correlation-based detection with integration:
# When bit=1: correlation = A1 * integral(cos^2) / fs = A1 * t_bit / 2
# When bit=0: correlation = 0
# Optimal threshold = A1 * t_bit / 4
correlation_threshold = A1 * t_bit / 5 # Adjusted threshold for better performance

# Eb/N0 Range
EbN0_dB = np.arange(0, 21, 1)  # 0 to 20 dB
BER = np.zeros(len(EbN0_dB))   # To store BER values

# Arrays to store constellation points for plotting
tx_constellation = []
rx_constellation = []

# Main Simulation Loop
for idx, EbN0 in enumerate(EbN0_dB):
    bask_signal = []
    for bit in sig: # BASK Modulation
        if bit == 1:
            a = A1 * np.cos(2 * np.pi * fc * t)  # High amplitude carrier for bit 1
        else:
            a = A0 * np.cos(2 * np.pi * fc * t)  # Zero amplitude (no carrier) for bit 0
        bask_signal.extend(a)
    bask_signal = np.array(bask_signal)  # Convert to numpy array for processing

    # Convert Eb/N0 from dB to linear scale
    EbN0_linear = 10**(EbN0 / 10)

    # Compute noise variance
    N0 = Eb / EbN0_linear
    # For discrete-time AWGN: sigma^2 = N0 * fs / 2
    # This accounts for the sampling frequency in discrete-time simulation
    noise_variance = N0 * fs / 2
    noise_std = np.sqrt(noise_variance)

    # Add AWGN (Additive White Gaussian Noise)
    noise = noise_std * np.random.randn(len(bask_signal))  # Generate Gaussian noise
    received = bask_signal + noise  # Simulate noisy channel

    # Demodulation and constellation collection
    demod = np.zeros(len(sig))  # Array to store demodulated bits
    tx_symbols = []  # Store transmitted constellation points
    rx_symbols = []  # Store received constellation points
    
    for i in range(len(sig)):  # Process each bit period
        start = i * samples_per_bit  # Start index for current bit
        end = (i + 1) * samples_per_bit  # End index for current bit
        
        # Transmitted symbol (for constellation)
        tx_segment = bask_signal[start:end]
        tx_symbol = np.sum(tx_segment * reference) / fs  # Integrate and normalize
        tx_symbols.append(tx_symbol)
        
        # Received symbol (for constellation) 
        rx_segment = received[start:end]
        rx_symbol = np.sum(rx_segment * reference) / fs  # Integrate and normalize
        rx_symbols.append(rx_symbol)
        
        # Decision using correlation threshold
        demod[i] = 1 if rx_symbol > correlation_threshold else 0

    # Store constellation points for the middle Eb/N0 value (10 dB)
    if EbN0 == 10:
        tx_constellation = tx_symbols[:10000]  # First 10000 symbols
        rx_constellation = rx_symbols[:10000]

    # BER Calculation
    bit_errors = np.sum(sig != demod)  # Count mismatched bits
    BER[idx] = bit_errors / num_bits  # Calculate bit error rate
    print(f"BASK - Eb/N0 = {EbN0} dB → BER = {BER[idx]:.6e} → Bit Errors = {bit_errors}")

# ================== SCIPY-BASED AM MODULATION/DEMODULATION ==================
print("\n--- Starting Scipy-based AM Modulation ---")

# Parameters for scipy AM
carrier_freq = fc
mod_index = 1.0  # Modulation index for AM
BER_scipy = np.zeros(len(EbN0_dB))

# Create message signal from bits (NRZ encoding)
message_bits = sig[:20]  # Use first 20 bits for comparison
message_signal = []
for bit in message_bits:
    if bit == 1:
        message_signal.extend([A1] * samples_per_bit)  # Positive amplitude for bit 1
    else:
        message_signal.extend([-A1] * samples_per_bit)  # Negative amplitude for bit 0 (Bipolar NRZ)
message_signal = np.array(message_signal)  # Convert to numpy array

# AM Modulation using scipy
t_total = np.arange(0, len(message_signal)) / fs  # Time vector for entire message
carrier = np.cos(2 * np.pi * carrier_freq * t_total)  # Carrier signal

# Standard AM modulation: s(t) = [1 + m(t)] * cos(2πfct)
am_signal = (1 + mod_index * message_signal / A1) * carrier  # Amplitude modulation

# Add noise and demodulate for BER calculation (using middle SNR)
EbN0_test = 10  # Test at 10 dB Eb/N0
EbN0_linear = 10**(EbN0_test / 10)  # Convert dB to linear scale
N0 = Eb / EbN0_linear  # Noise power spectral density
noise_variance = N0 * fs  # Noise variance for discrete-time
noise_std = np.sqrt(noise_variance)  # Noise standard deviation
noise = noise_std * np.random.randn(len(am_signal))  # Generate AWGN
am_received = am_signal + noise  # Add noise to AM signal

# AM Demodulation using envelope detection (Hilbert transform)
analytic_signal = hilbert(am_received)  # Compute analytic signal
envelope = np.abs(analytic_signal)  # Extract envelope (magnitude)
# Remove DC component and recover message
demod_message = envelope - 1  # Remove DC bias from envelope

# Store signals for plotting
scipy_tx_signal = am_signal  # Store transmitted AM signal
scipy_rx_signal = am_received  # Store received AM signal (with noise)

print(f"Scipy AM - Signal length: {len(scipy_tx_signal)} samples")

# Plot comparison of BASK and Scipy AM signals
plt.figure(1, figsize=(15, 10))

# Time vector for 20 bits (each bit has duration t_bit)
time_20_bits = np.arange(0, 20*t_bit, 1/fs)

# Subplot 1: BASK Signals
plt.subplot(2, 1, 1)
plt.plot(time_20_bits, bask_signal[:20*samples_per_bit], 'k', linewidth=2, label='BASK TX Signal')
plt.plot(time_20_bits, received[:20*samples_per_bit], 'g', alpha=0.4, linewidth=1, label='BASK RX Signal')

# Display the first 20 bits being transmitted
print(f"First 20 bits: {sig_20}")
plt.title(f"BASK Modulation - Transmitted and Received Signals\nFirst 20 bits: {sig_20}")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Add vertical lines to show bit boundaries
for i in range(1, 20):
    plt.axvline(x=i*t_bit, color='red', linestyle='--', alpha=0.3)

# Subplot 2: Scipy AM Signals
plt.subplot(2, 1, 2)
time_scipy = np.arange(0, len(scipy_tx_signal)) / fs
plt.plot(time_scipy, scipy_tx_signal, 'b', linewidth=2, label='Scipy AM TX Signal')
plt.plot(time_scipy, scipy_rx_signal, 'r', alpha=0.4, linewidth=1, label='Scipy AM RX Signal')

plt.title(f"Scipy AM Modulation - Transmitted and Received Signals\nFirst 20 bits: {sig_20}")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Add vertical lines to show bit boundaries
for i in range(1, 20):
    plt.axvline(x=i*t_bit, color='red', linestyle='--', alpha=0.3)

plt.tight_layout()

# Plot BER vs Eb/N0 comparison
plt.figure(2)
plt.semilogy(EbN0_dB, BER, 'o-', linewidth=2, label='Custom BASK Implementation') # Used for Log plots

# Add theoretical BASK BER curve for comparison
EbN0_linear_theory = 10**(EbN0_dB / 10)
# Theoretical BER for coherent BASK: BER = 0.5 * erfc(sqrt(Eb/N0))
from scipy.special import erfc
BER_theory = 0.5 * erfc(np.sqrt(EbN0_linear_theory))
plt.semilogy(EbN0_dB, BER_theory, '--', linewidth=2, label='Theoretical BASK BER')

plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs $E_b/N_0$ Comparison")
plt.legend()
plt.grid(True, which='both')

# Plot constellation diagrams
plt.figure(3, figsize=(16, 8))
plt.suptitle("BASK Constellation Diagrams", fontsize=16, fontweight='bold')

# Transmitted constellation
plt.subplot(1, 2, 1)
tx_0_bits = [tx_constellation[i] for i in range(len(tx_constellation)) if sig[i] == 0]
tx_1_bits = [tx_constellation[i] for i in range(len(tx_constellation)) if sig[i] == 1]

# Add some vertical spread to make points more visible
np.random.seed(42)  # For reproducible jitter pattern
jitter_0 = np.random.normal(0, 0.05, len(tx_0_bits))  # Vertical jitter for bit 0 points
jitter_1 = np.random.normal(0, 0.05, len(tx_1_bits))  # Vertical jitter for bit 1 points

plt.scatter(tx_0_bits, jitter_0, c='darkblue', alpha=0.7, label=f'Bit 0', s=25, marker='o')
plt.scatter(tx_1_bits, jitter_1, c='darkred', alpha=0.7, label=f'Bit 1', s=25, marker='^')

plt.xlabel('In-phase Component', fontsize=12, fontweight='bold')
plt.ylabel('Quadrature Component', fontsize=12, fontweight='bold')
plt.title('Transmitted Constellation (10,000 symbols)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(-0.3, 0.3)
plt.axhline(y=0, color='black', linewidth=1, alpha=0.5)
plt.axvline(x=0, color='black', linewidth=1, alpha=0.5)

# Received constellation
plt.subplot(1, 2, 2)
rx_0_bits = [rx_constellation[i] for i in range(len(rx_constellation)) if sig[i] == 0]
rx_1_bits = [rx_constellation[i] for i in range(len(rx_constellation)) if sig[i] == 1]

# Add some vertical spread to make points more visible
jitter_rx_0 = np.random.normal(0, 0.05, len(rx_0_bits))  # Vertical jitter for received bit 0 points
jitter_rx_1 = np.random.normal(0, 0.05, len(rx_1_bits))  # Vertical jitter for received bit 1 points

plt.scatter(rx_0_bits, jitter_rx_0, c='lightblue', alpha=0.5, label=f'Received Bit 0', s=15, marker='o')
plt.scatter(rx_1_bits, jitter_rx_1, c='lightcoral', alpha=0.5, label=f'Received Bit 1', s=15, marker='^')

# Decision threshold
plt.axvline(x=correlation_threshold, color='green', linestyle='--', linewidth=3, 
           label=f'Decision Threshold = {correlation_threshold:.2f}', zorder=4)

# Add decision regions
plt.axvspan(-10, correlation_threshold, alpha=0.1, color='blue')
plt.axvspan(correlation_threshold, 10, alpha=0.1, color='red')

plt.xlabel('In-phase Component', fontsize=12, fontweight='bold')
plt.ylabel('Quadrature Component', fontsize=12, fontweight='bold')
plt.title('Received Constellation (Eb/N0 = 10 dB, 10,000 symbols)', fontsize=14, fontweight='bold')
plt.legend(fontsize=9, loc='upper right')
plt.grid(True, alpha=0.3)
plt.ylim(-0.3, 0.3)
plt.axhline(y=0, color='black', linewidth=1, alpha=0.5)
plt.axvline(x=0, color='black', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.show()