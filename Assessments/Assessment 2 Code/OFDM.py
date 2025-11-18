import math
import numpy as np
from scipy.fftpack import fft
from scipy.special import erfc
import matplotlib.pyplot as plt
# ================== OFDM SYSTEM PARAMETERS ==================
print("OFDM System Parameters")
print("=" * 60)

# Frequency domain parameters
bandwidth = 10000000          # 10 MHz - Total allocated bandwidth
FFT_Size = 512                # 512-point FFT
active_subcarriers = 480      # 480 active subcarriers (480/512 = 93.75% utilization)
subcarrier_spacing = 15000    # 15 kHz - Spacing between subcarriers
sampling_rate = 7680000       # 7.68 MHz - Sampling frequency

# Time domain parameters (derived from frequency parameters)
T_useful = 1 / subcarrier_spacing                    # Useful OFDM symbol duration (no CP)
cp_length = 0.125                                     # Cyclic prefix ratio (1/8)
T_CP = T_useful * cp_length                          # CP duration
T_symbol = T_useful + T_CP                           # Total OFDM symbol duration (with CP)
cp_length_samples = int(cp_length * FFT_Size)        # CP length in samples

# Modulation parameters
modulation = "QPSK"           # QPSK modulation
A = 5                         # Amplitude for constellation points
bits_per_symbol = 2           # QPSK carries 2 bits per symbol

# SNR range for simulation
snr = range(0, 21, 1)         # SNR from 0 to 20 dB

# ================== PARAMETER VALIDATION ==================
# Verify that parameters are consistent
calculated_sampling_rate = FFT_Size * subcarrier_spacing
occupied_bandwidth = active_subcarriers * subcarrier_spacing
guard_bandwidth = bandwidth - occupied_bandwidth

print(f"Bandwidth: {bandwidth/1e6:.2f} MHz")
print(f"FFT Size: {FFT_Size}")
print(f"Active Subcarriers: {active_subcarriers} ({active_subcarriers/FFT_Size*100:.1f}% utilization)")
print(f"Null Subcarriers: {FFT_Size - active_subcarriers} (DC + guards)")
print(f"Subcarrier Spacing: {subcarrier_spacing/1e3:.1f} kHz")
print(f"Sampling Rate: {sampling_rate/1e6:.2f} MHz")
print(f"Calculated Sampling Rate: {calculated_sampling_rate/1e6:.2f} MHz", end="")
    
print(f"\nOccupied Bandwidth: {occupied_bandwidth/1e6:.2f} MHz")
print(f"Guard Bandwidth: {guard_bandwidth/1e6:.2f} MHz ({guard_bandwidth/bandwidth*100:.1f}%)")

print(f"\nTiming Parameters:")
print(f"Useful Symbol Duration (T_useful): {T_useful*1e6:.2f} μs")
print(f"CP Duration (T_CP): {T_CP*1e6:.2f} μs ({cp_length*100:.1f}% of useful)")
print(f"Total Symbol Duration: {T_symbol*1e6:.2f} μs")
print(f"CP Length (samples): {cp_length_samples}")

print(f"\nData Rate Calculation:")
print(f"Symbols per OFDM symbol: {active_subcarriers}")
print(f"Bits per OFDM symbol: {active_subcarriers * bits_per_symbol}")
print(f"OFDM symbol rate: {1/T_symbol:.2f} symbols/s")
print(f"Theoretical data rate: {active_subcarriers * bits_per_symbol / T_symbol / 1e6:.2f} Mbps")
print("=" * 60)
print()

"""
Transmitter (per-user / central scheduler)
message bits -> modulation -> serial to parallel -> IFFT -> add CP -> parallel to serial -> transmit

Receiver
receive signal -> serial to parallel -> remove CP -> FFT -> parallel to serial -> demodulation

1. Modulation → Complex symbols (I+jQ)
2. S/P → Reshape to OFDM blocks
3. Map to subcarriers → Pad with zeros for unused carriers
4. IFFT → Convert to time domain (complex signal)
5. Add CP → Copy last samples to beginning
6. Transmit → Add complex AWGN noise
7. Remove CP → Discard cyclic prefix
8. FFT → Convert back to frequency domain
9. Extract subcarriers → Get active carriers
10. Demodulate → Simple symbol slicing (NO correlation!)
11. P/S → Flatten back to symbol stream

"""

# Generate random bits
num_bits = 100000
bits = np.random.randint(0, 2, num_bits)

# Ensure even number of bits (for QPSK - 2 bits per symbol)
if num_bits % 2 != 0:
    bits = bits[:-1]
    num_bits = len(bits)

# Map bits to QPSK symbols
symbols = bits[::2] * 2 + bits[1::2]  # Group bits into pairs for QPSK
num_symbols = len(symbols)

# Pad symbols to make them divisible by active_subcarriers
symbols_needed = int(np.ceil(num_symbols / active_subcarriers) * active_subcarriers)
symbols_to_pad = symbols_needed - num_symbols

if symbols_to_pad > 0:
    symbols = np.concatenate([symbols, np.zeros(symbols_to_pad, dtype=int)])
    print(f"Padded {symbols_to_pad} symbols to create complete OFDM blocks")

num_ofdm_symbols = symbols_needed // active_subcarriers
print(f"Total QPSK symbols: {num_symbols}")
print(f"OFDM symbols to transmit: {num_ofdm_symbols}")
print(f"Symbols per OFDM symbol: {active_subcarriers}")
print()

# Map symbols to complex constellation
constellation_phases = {
    0: np.pi/4,      # 00 -> 45°
    1: 3*np.pi/4,    # 01 -> 135°
    2: -np.pi/4,     # 10 -> 315°
    3: -3*np.pi/4    # 11 -> 225°
}

# Create I and Q constellation points
constellation_I = {k: A * np.cos(phase) for k, phase in constellation_phases.items()}  # In-phase components
constellation_Q = {k: A * np.sin(phase) for k, phase in constellation_phases.items()}  # Quadrature components
qpsk_modulated = np.array([constellation_I[s] + 1j * constellation_Q[s] for s in symbols])

# Serial to Parallel (now guaranteed to work since we padded)
ofdm_symbols = qpsk_modulated.reshape(-1, active_subcarriers)  # Shape: (num_ofdm_symbols, 480)

# Map to FFT bins
fft_input = np.zeros((ofdm_symbols.shape[0], FFT_Size), dtype=complex) # Initialize with zeros
start = (FFT_Size - active_subcarriers)//2
end = start + active_subcarriers
fft_input[:, start:end] = ofdm_symbols  # Map active subcarriers to center of FFT bins

# IFFT to get time-domain OFDM symbols
ofdm_time_domain = np.fft.ifft(fft_input, n=FFT_Size, axis=1)  # 512 FFT 

# Add Cyclic Prefix (using pre-calculated cp_length_samples from parameters)
ofdm_with_cp = np.hstack((ofdm_time_domain[:, -cp_length_samples:], ofdm_time_domain))

BER_ofdm = np.zeros(len(snr))  # Store BER for each SNR value

for snr_db in snr:
    # compute average time-domain power of transmitted waveform (including CP)
    tx_waveform = ofdm_with_cp.reshape(-1)  # serial time-domain samples
    signal_power = np.mean(np.abs(tx_waveform)**2)  # average sample power

    SNR_linear = 10**(snr_db/10)
    noise_power_per_complex_sample = signal_power / SNR_linear     # Esample / SNR
    # for complex AWGN, each of real and imag has variance = noise_power_per_complex_sample/2
    noise_variance = np.sqrt(noise_power_per_complex_sample/2)

    # generate complex AWGN of same shape as ofdm_with_cp
    noise = noise_variance * (np.random.randn(*ofdm_with_cp.shape) + 1j*np.random.randn(*ofdm_with_cp.shape))

    received_signal = ofdm_with_cp + noise # Received signal with noise

    # Remove CP
    received_signal_no_cp = received_signal[:, cp_length_samples:]

    # FFT to convert back to frequency domain (No need for correlation)
    received_ofdm_symbols = np.fft.fft(received_signal_no_cp, n=FFT_Size, axis=1)

    # Extract active subcarriers
    received_active_subcarriers = received_ofdm_symbols[:, start:end].flatten()

    # Demodulation (Symbol slicing)

    demod_symbols = []
    for rx_symbol in received_active_subcarriers:
        distances = [abs(rx_symbol - (constellation_I[s] + 1j * constellation_Q[s])) for s in range(4)]
        demod_symbols.append(np.argmin(distances))

    demod_symbols = np.array(demod_symbols)

    # Convert symbols back to bits
    bit_pairs = {
    0: [0,0],
    1: [0,1],
    2: [1,0],
    3: [1,1]
    }

    demod_bits = np.array([b for s in demod_symbols for b in bit_pairs[s]])
    demod_bits = demod_bits[:num_bits]  # Trim to original number of bits
    # Calculate Bit Error Rate (BER)
    num_bit_errors = np.sum(bits != demod_bits)
    ber = num_bit_errors / num_bits
    BER_ofdm[snr_db] = ber  # Store instead of just printing
    print(f"Bit Error Rate (BER): {ber}")

    # After demodulation, store for plotting at specific SNR
    if snr_db == 10:
        tx_constellation = qpsk_modulated[:1000]
        rx_constellation = received_active_subcarriers[:1000]


plt.figure(1)
plt.plot(snr, BER_ofdm, 'o-', label='Simulated BER')
plt.yscale('log')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('OFDM QPSK BER vs SNR')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(2)
plt.plot(snr, BER_ofdm, 'o-', label='Simulated BER')
plt.plot(snr, 0.5 * erfc(np.sqrt(10**(np.array(snr)/10))), 'r--', label='Theoretical BER')
plt.yscale('log')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('OFDM QPSK BER vs SNR')
plt.grid(True)
plt.legend()
plt.show()