import math
import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
# ================== OFDM SYSTEM PARAMETERS ==================
print("OFDM System Parameters")
print("=" * 60)

# Frequency domain parameters (all given in brief)
bandwidth = 10000000          # 10 MHz - Total allocated bandwidth
FFT_Size = 512                # 512-point FFT
active_subcarriers = 480      # 480 active subcarriers (480/512 = 93.75% utilisation)
subcarrier_spacing = 15000    # 15 kHz - Spacing between subcarriers
sampling_rate = 7680000       # 7.68 MHz - Sampling frequency

# Time domain parameters (derived from frequency parameters)
T_useful = 1 / subcarrier_spacing                    # Useful OFDM symbol duration (no CP)
cp_length = 0.125                                    # Cyclic prefix ratio (1/8)
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

# ================== STAGE 1: Generate Random Bits ==================
num_bits = 100000
bits = np.random.randint(0, 2, num_bits)

# Ensure even number of bits (for QPSK - 2 bits per symbol)
if num_bits % 2 != 0:
    bits = bits[:-1]
    num_bits = len(bits)

# ================== PLOT 1: Original Bit Sequence ==================
plt.figure(figsize=(12, 3))
plt.stem(bits[:100], linefmt='b-', markerfmt='bo', basefmt=' ')
plt.xlabel('Bit Index')
plt.ylabel('Bit Value')
plt.title('Stage 1: Original Bit Sequence (First 100 bits)')
plt.grid(True, alpha=0.3)
plt.ylim([-0.2, 1.2])
plt.tight_layout()
plt.savefig("Step-1-Original-Sequence.png")
plt.show()

# ================== STAGE 2: QPSK Modulation ==================
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

# ================== PLOT 2: QPSK Constellation ==================
# Colors and markers for different symbols
symbol_colors = ['blue', 'red', 'green', 'orange']
symbol_labels = ['00', '01', '10', '11']
symbol_markers = ['o', 's', '^', 'D']

plt.figure(figsize=(8, 8))
# Plot transmitted symbols with different colors for each symbol type
for sym in range(4):
    sym_indices = np.where(symbols[:1000] == sym)[0]
    sym_I = [qpsk_modulated[i].real for i in sym_indices]
    sym_Q = [qpsk_modulated[i].imag for i in sym_indices]
    plt.scatter(sym_I, sym_Q, c=symbol_colors[sym], alpha=0.6, s=20, 
               marker=symbol_markers[sym], label=f'{symbol_labels[sym]}', 
               edgecolors='white', linewidth=0.5)

# Plot ideal constellation points
for s, phase in constellation_phases.items():
    I = constellation_I[s]
    Q = constellation_Q[s]
    plt.scatter(I, Q, c=symbol_colors[s], s=200, marker=symbol_markers[s], 
               edgecolors='black', linewidth=2, zorder=5)
    plt.annotate(f'{symbol_labels[s]}', (I, Q), xytext=(10, 10), 
                textcoords='offset points', fontsize=12, fontweight='bold',
                color='black', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='yellow', alpha=0.7))

plt.xlabel('In-Phase (I)', fontsize=12, fontweight='bold')
plt.ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
plt.title('Stage 2: QPSK Constellation Mapping\n(First 1000 symbols)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.legend(loc='upper right', framealpha=0.9, fontsize=10)
plt.tight_layout()
plt.savefig("Step-2-QPSK-Mapping.png")
plt.show()

# ================== STAGE 3: Serial to Parallel & Subcarrier Mapping ==================
# Serial to Parallel
ofdm_symbols = qpsk_modulated.reshape(-1, active_subcarriers)  # Shape: (num_ofdm_symbols, 480)

# Map to FFT bins with DC null and guard bands
# Standard OFDM subcarrier allocation:
# - Lower guard band: indices 0 to 15 (16 carriers)
# - Lower active: indices 16 to 255 (240 carriers)
# - DC null: index 256 (1 carrier - center frequency)
# - Upper active: indices 257 to 496 (240 carriers)
# - Upper guard band: indices 497 to 511 (15 carriers)
fft_input = np.zeros((ofdm_symbols.shape[0], FFT_Size), dtype=complex) # Initialize with zeros

# Split active subcarriers into lower and upper halves (240 each)
lower_half = active_subcarriers // 2  # 240
upper_half = active_subcarriers - lower_half  # 240

# Map lower half (before DC)
lower_start = 16  # After lower guard band
lower_end = lower_start + lower_half  # 16 + 240 = 256
fft_input[:, lower_start:lower_end] = ofdm_symbols[:, :lower_half]

# Index 256 remains null (DC subcarrier)

# Map upper half (after DC)
upper_start = 257  # After DC null
upper_end = upper_start + upper_half  # 257 + 240 = 497
fft_input[:, upper_start:upper_end] = ofdm_symbols[:, lower_half:]

# For receiver (define indices for extraction)
start_lower = lower_start
end_lower = lower_end
start_upper = upper_start
end_upper = upper_end

# ================== PLOT 3: Frequency Domain Subcarrier Allocation ==================
plt.figure(figsize=(14, 5))
plt.subplot(2, 1, 1)
plt.stem(np.abs(fft_input[0, :]), linefmt='b-', markerfmt='bo', basefmt=' ')
plt.xlabel('Subcarrier Index')
plt.ylabel('Magnitude')
plt.title('Stage 3: Frequency Domain - Subcarrier Allocation (First OFDM Symbol)')
plt.grid(True, alpha=0.3)
# Lower guard band
plt.axvspan(0, lower_start, alpha=0.2, color='red', label='Guard bands')
# Upper guard band
plt.axvspan(end_upper, FFT_Size, alpha=0.2, color='red')
# Lower active carriers
plt.axvspan(lower_start, end_lower, alpha=0.2, color='green', label='Active carriers')
# DC null
plt.axvline(x=256, color='orange', linewidth=2, linestyle='--', label='DC null')
# Upper active carriers
plt.axvspan(start_upper, end_upper, alpha=0.2, color='green')
plt.legend()

plt.subplot(2, 1, 2)
plt.stem(np.angle(fft_input[0, :]), linefmt='g-', markerfmt='go', basefmt=' ')
plt.xlabel('Subcarrier Index')
plt.ylabel('Phase (radians)')
plt.title('Phase of Subcarriers')
plt.grid(True, alpha=0.3)
# Lower guard band
plt.axvspan(0, lower_start, alpha=0.2, color='red')
# Upper guard band
plt.axvspan(end_upper, FFT_Size, alpha=0.2, color='red')
# Lower active carriers
plt.axvspan(lower_start, end_lower, alpha=0.2, color='green')
# DC null
plt.axvline(x=256, color='orange', linewidth=2, linestyle='--', label='DC null')
# Upper active carriers
plt.axvspan(start_upper, end_upper, alpha=0.2, color='green')
plt.legend()
plt.tight_layout()
plt.savefig("Step-3-Frequency-Domain-Subcarrier-Allocation.png")
plt.show()

# ================== STAGE 4: IFFT & Cyclic Prefix Addition ==================
# IFFT to get time-domain OFDM symbols
ofdm_time_domain = np.fft.ifft(fft_input, n=FFT_Size, axis=1)  # 512 FFT 

# ================== PLOT 4: Time Domain OFDM Signal (Before CP) ==================
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(np.real(ofdm_time_domain[0, :]), 'b-', label='Real (I)')
plt.plot(np.imag(ofdm_time_domain[0, :]), 'r-', label='Imaginary (Q)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Stage 4a: Time Domain OFDM Signal - After IFFT (First Symbol, No CP)')
plt.grid(True, alpha=0.3)
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(np.abs(ofdm_time_domain[0, :]), 'g-')
plt.xlabel('Sample Index')
plt.ylabel('Magnitude')
plt.title('Magnitude of Time Domain Signal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Step-4a-Time-Domain-OFDM-Signal-After-IFFT.png")
plt.show()

# Add Cyclic Prefix (using pre-calculated cp_length_samples from parameters)
ofdm_with_cp = np.hstack((ofdm_time_domain[:, -cp_length_samples:], ofdm_time_domain))

# ================== PLOT 5: Time Domain OFDM Signal (With CP) ==================
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(np.real(ofdm_with_cp[0, :]), 'b-', label='Real (I)')
plt.plot(np.imag(ofdm_with_cp[0, :]), 'r-', label='Imaginary (Q)')
plt.axvspan(0, cp_length_samples, alpha=0.3, color='yellow', label='Cyclic Prefix')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title(f'Stage 4b: Time Domain OFDM Signal - With Cyclic Prefix (First Symbol, CP={cp_length_samples} samples)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.abs(ofdm_with_cp[0, :]), 'g-')
plt.axvspan(0, cp_length_samples, alpha=0.3, color='yellow')
plt.xlabel('Sample Index')
plt.ylabel('Magnitude')
plt.title('Magnitude of Time Domain Signal with CP')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Step-4b-Time-Domain-OFDM-Signal-After-IFFT-With-64-CP-Samples.png")
plt.show()

# ================== STAGE 5: Transmission through AWGN Channel ==================
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

    # ================== STAGE 6: Receiver Processing ==================
    # Remove CP
    received_signal_no_cp = received_signal[:, cp_length_samples:]

    # FFT to convert back to frequency domain (No need for correlation)
    received_ofdm_symbols = np.fft.fft(received_signal_no_cp, n=FFT_Size, axis=1)

    # Extract active subcarriers (lower and upper halves, excluding DC null)
    received_lower = received_ofdm_symbols[:, start_lower:end_lower]  # Lower 240 carriers
    received_upper = received_ofdm_symbols[:, start_upper:end_upper]  # Upper 240 carriers
    received_active_subcarriers = np.hstack((received_lower, received_upper)).flatten()  # Combine and flatten

    # ================== STAGE 7: Demodulation & BER Calculation ==================
    # Demodulation (Symbol slicing)
    demod_symbols = []

    # Broadcasting used to improve efficiency
    const_points = np.array([constellation_I[s] + 1j*constellation_Q[s] for s in range(4)])

    distances = abs(received_active_subcarriers[:,None] - const_points[None,:])
    demod_symbols = np.argmin(distances, axis=1)

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
    print(f"Bit Error Rate (BER): {ber} for SNR = {snr_db} dB")

    # After demodulation, store for plotting at specific SNR
    if snr_db == 10:
        tx_constellation = qpsk_modulated[:1000]
        rx_constellation = received_active_subcarriers[:1000]
        
        # ================== PLOT 6: Received Signal in Frequency Domain ==================
        plt.figure(figsize=(14, 5))
        plt.subplot(2, 1, 1)
        plt.stem(np.abs(received_ofdm_symbols[0, :]), linefmt='b-', markerfmt='bo', basefmt=' ')
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Magnitude')
        plt.title(f'Stage 5: Received Signal in Frequency Domain (After FFT, SNR={snr_db} dB)')
        plt.grid(True, alpha=0.3)
        # Lower guard band
        plt.axvspan(0, lower_start, alpha=0.2, color='red', label='Guard bands')
        # Upper guard band
        plt.axvspan(end_upper, FFT_Size, alpha=0.2, color='red')
        # Lower active carriers
        plt.axvspan(lower_start, end_lower, alpha=0.2, color='green', label='Active carriers')
        # DC null
        plt.axvline(x=256, color='orange', linewidth=2, linestyle='--', label='DC null')
        # Upper active carriers
        plt.axvspan(start_upper, end_upper, alpha=0.2, color='green')
        plt.legend(loc='upper left')
        
        plt.subplot(2, 1, 2)
        plt.stem(np.angle(received_ofdm_symbols[0, :]), linefmt='g-', markerfmt='go', basefmt=' ')
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Phase (radians)')
        plt.title('Phase of Received Subcarriers')
        plt.grid(True, alpha=0.3)
        # Lower guard band
        plt.axvspan(0, lower_start, alpha=0.2, color='red')
        # Upper guard band
        plt.axvspan(end_upper, FFT_Size, alpha=0.2, color='red')
        # Lower active carriers
        plt.axvspan(lower_start, end_lower, alpha=0.2, color='green')
        # DC null
        plt.axvline(x=256, color='orange', linewidth=2, linestyle='--', label='DC null')
        # Upper active carriers
        plt.axvspan(start_upper, end_upper, alpha=0.2, color='green')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig("Step-5-Frequency-Domain-Received-Signal.png")
        plt.show()
        
        # ================== PLOT 7: Constellation Comparison ==================
        plt.figure(figsize=(14, 6))
        
        # Transmitted constellation
        plt.subplot(1, 2, 1)
        for sym in range(4):
            sym_indices = np.where(symbols[:1000] == sym)[0]
            sym_I = [tx_constellation[i].real for i in sym_indices if i < len(tx_constellation)]
            sym_Q = [tx_constellation[i].imag for i in sym_indices if i < len(tx_constellation)]
            plt.scatter(sym_I, sym_Q, c=symbol_colors[sym], alpha=0.6, s=20, 
                       marker=symbol_markers[sym], label=f'{symbol_labels[sym]}', 
                       edgecolors='white', linewidth=0.5)
        
        # Plot ideal constellation points
        for s, phase in constellation_phases.items():
            I = constellation_I[s]
            Q = constellation_Q[s]
            plt.scatter(I, Q, c=symbol_colors[s], s=200, marker=symbol_markers[s], 
                       edgecolors='black', linewidth=2, zorder=5)
        
        plt.xlabel('In-Phase (I)', fontsize=12, fontweight='bold')
        plt.ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
        plt.title('Transmitted Constellation', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        # Received constellation with decision regions
        plt.subplot(1, 2, 2)
        
        # Add colored decision regions (quadrants)
        plt.fill([0, 10, 10, 0], [0, 0, 10, 10], alpha=0.1, color=symbol_colors[0])  # Quadrant I (00)
        plt.fill([-10, 0, 0, -10], [0, 0, 10, 10], alpha=0.1, color=symbol_colors[1])  # Quadrant II (01)
        plt.fill([-10, 0, 0, -10], [-10, -10, 0, 0], alpha=0.1, color=symbol_colors[3])  # Quadrant III (11)
        plt.fill([0, 10, 10, 0], [-10, -10, 0, 0], alpha=0.1, color=symbol_colors[2])  # Quadrant IV (10)
        
        for sym in range(4):
            sym_indices = np.where(symbols[:1000] == sym)[0]
            sym_I = [rx_constellation[i].real for i in sym_indices if i < len(rx_constellation)]
            sym_Q = [rx_constellation[i].imag for i in sym_indices if i < len(rx_constellation)]
            plt.scatter(sym_I, sym_Q, c=symbol_colors[sym], alpha=0.4, s=15, 
                       marker=symbol_markers[sym], label=f'{symbol_labels[sym]}')
        
        # Plot ideal constellation points
        for s, phase in constellation_phases.items():
            I = constellation_I[s]
            Q = constellation_Q[s]
            plt.scatter(I, Q, c=symbol_colors[s], s=200, marker=symbol_markers[s], 
                       edgecolors='black', linewidth=2, zorder=5)
        
        plt.xlabel('In-Phase (I)', fontsize=12, fontweight='bold')
        plt.ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
        plt.title(f'Received Constellation (SNR={snr_db} dB)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        plt.suptitle('Stage 6: Constellation Comparison - Transmitted vs Received', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("Step-6-Tx-Vs-Rx-Constellation-Diagrams.png")
        plt.show()


# ================== PLOT 8: BER Performance (Simulated Only) ==================
plt.figure(figsize=(10, 6))
plt.plot(snr, BER_ofdm, 'o-', linewidth=2, markersize=8, label='Simulated BER')
plt.yscale('log')
plt.ylim([1e-6, 1])
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Bit Error Rate (BER)', fontsize=12)
plt.title('Stage 7: OFDM QPSK BER vs SNR Performance', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Step-7-OFDM-BER-Just-Simulated.png")
plt.show()

# ================== PLOT 9: BER Performance with Theoretical Comparison ==================
plt.figure(figsize=(10, 6))
plt.plot(snr, BER_ofdm, 'o-', linewidth=2, markersize=8, label='Simulated BER')
# Theoretical BER for QPSK in AWGN
# Pb = Q * sqrt(2*Eb/N0) = 0.5 * erfc(sqrt(Eb/N0))
plt.plot(snr, 0.5 * erfc(np.sqrt(10**(np.array(snr)/10))), 'r--', linewidth=2, label='Theoretical BER (QPSK AWGN)')
plt.yscale('log')
plt.ylim([1e-6, 1])
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Bit Error Rate (BER)', fontsize=12)
plt.title('Stage 7: OFDM QPSK BER Performance - Simulated vs Theoretical', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Step-7-OFDM-BER-Sim-Theoretical.png")
plt.show()