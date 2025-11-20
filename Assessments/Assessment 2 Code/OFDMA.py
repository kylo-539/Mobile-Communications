import math
import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

# ================== OFDM SYSTEM PARAMETERS ==================
print("OFDM System Parameters")
print("=" * 60)

bandwidth = 10000000
FFT_Size = 512
active_subcarriers = 480
subcarrier_spacing = 15000
sampling_rate = 7680000

T_useful = 1 / subcarrier_spacing
cp_length = 0.125
T_CP = T_useful * cp_length
T_symbol = T_useful + T_CP
cp_length_samples = int(cp_length * FFT_Size)

modulation = "QPSK"
A = 5
bits_per_symbol = 2

snr = range(0, 21, 1)

# ================== PARAMETER VALIDATION ==================
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

# ================== STAGE 1: Generate Random Bits for Each User ==================
num_bits = 100000
user1_bits = np.random.randint(0, 2, num_bits)
user2_bits = np.random.randint(0, 2, num_bits)

if num_bits % 2 != 0:
    user1_bits = user1_bits[:-1]
    user2_bits = user2_bits[:-1]
    num_bits = len(user1_bits)

# ================== PLOT 1: Original Bit Sequence ==================
plt.figure(figsize=(12, 3))
plt.stem(user1_bits[:100], linefmt='b-', markerfmt='bo', basefmt=' ')
plt.stem(user2_bits[:100], linefmt='r-', markerfmt='ro', basefmt=' ')
plt.xlabel('Bit Index')
plt.ylabel('Bit Value')
plt.legend(['User 1 Bits', 'User 2 Bits'])
plt.title('Stage 1: Original Bit Sequence (First 100 bits)')
plt.grid(True, alpha=0.3)
plt.ylim([-0.2, 1.2])
plt.tight_layout()
plt.savefig("OFDMA-Step-1-Original-Sequence.png")
plt.show()

# ================== STAGE 2: QPSK Modulation ==================
u1_symbols = user1_bits[::2] * 2 + user1_bits[1::2]
num_symbols_u1 = len(u1_symbols)

u2_symbols = user2_bits[::2] * 2 + user2_bits[1::2]
num_symbols_u2 = len(u2_symbols)

print(f"User 1 QPSK symbols: {num_symbols_u1}")
print(f"User 2 QPSK symbols: {num_symbols_u2}")
print()

constellation_phases = {
    0: np.pi/4,
    1: 3*np.pi/4,
    2: -np.pi/4,
    3: -3*np.pi/4
}

constellation_I = {k: A * np.cos(phase) for k, phase in constellation_phases.items()}
constellation_Q = {k: A * np.sin(phase) for k, phase in constellation_phases.items()}

symbols = np.concatenate([u1_symbols, u2_symbols])
qpsk_modulated = np.array([constellation_I[s] + 1j * constellation_Q[s] for s in symbols])

# ================== PLOT 2: QPSK Constellation ==================
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
plt.savefig("OFDMA-Step-2-QPSK-Mapping.png")
plt.show()

# ================== STAGE 3: Serial to Parallel & Subcarrier Mapping ==================
u1_qpsk_modulated = qpsk_modulated[:num_symbols_u1]
u2_qpsk_modulated = qpsk_modulated[num_symbols_u1:num_symbols_u1+num_symbols_u2]

symbols_per_user = 240

num_ofdm_symbols_u1 = int(np.ceil(len(u1_qpsk_modulated) / symbols_per_user))
num_ofdm_symbols_u2 = int(np.ceil(len(u2_qpsk_modulated) / symbols_per_user))
num_ofdm_symbols = max(num_ofdm_symbols_u1, num_ofdm_symbols_u2)

total_symbols_per_user = num_ofdm_symbols * symbols_per_user

if len(u1_qpsk_modulated) < total_symbols_per_user:
    padding = np.zeros(total_symbols_per_user - len(u1_qpsk_modulated), dtype=complex)
    u1_qpsk_modulated = np.concatenate([u1_qpsk_modulated, padding])

if len(u2_qpsk_modulated) < total_symbols_per_user:
    padding = np.zeros(total_symbols_per_user - len(u2_qpsk_modulated), dtype=complex)
    u2_qpsk_modulated = np.concatenate([u2_qpsk_modulated, padding])

print(f"Number of OFDM frames: {num_ofdm_symbols}")
print(f"Symbols per user per OFDM frame: {symbols_per_user}")
print(f"Total symbols per user (after padding): {total_symbols_per_user}")
print()

# Derive subcarrier allocation from FFT size and guard bands
lower_guard = 16
upper_guard = 16
dc_null = 1

lower_half = 240
upper_half = 240

lower_start = lower_guard
lower_end = lower_start + lower_half

dc_index = FFT_Size // 2

upper_start = dc_index + dc_null
upper_end = upper_start + upper_half

print(f"Subcarrier Allocation:")
print(f"Lower guard band: 0-{lower_guard-1} ({lower_guard} subcarriers)")
print(f"User 1 (lower): {lower_start}-{lower_end-1} ({lower_half} subcarriers)")
print(f"DC null: {dc_index}")
print(f"User 2 (upper): {upper_start}-{upper_end-1} ({upper_half} subcarriers)")
print(f"Upper guard band: {upper_end}-{FFT_Size-1} ({upper_guard} subcarriers)")
print()

fft_input = np.zeros((num_ofdm_symbols, FFT_Size), dtype=complex)

user1_ofdm_symbols = u1_qpsk_modulated.reshape(num_ofdm_symbols, symbols_per_user)
user2_ofdm_symbols = u2_qpsk_modulated.reshape(num_ofdm_symbols, symbols_per_user)

fft_input[:, lower_start:lower_end] = user1_ofdm_symbols[:, :]
fft_input[:, upper_start:upper_end] = user2_ofdm_symbols[:, :]

# ================== PLOT 3: Frequency Domain Subcarrier Allocation ==================
plt.figure(figsize=(14, 5))
plt.subplot(2, 1, 1)
plt.stem(np.abs(fft_input[0, :]), linefmt='b-', markerfmt='bo', basefmt=' ')
plt.xlabel('Subcarrier Index')
plt.ylabel('Magnitude')
plt.title('Stage 3: Frequency Domain - Subcarrier Allocation (First OFDM Symbol)')
plt.grid(True, alpha=0.3)
plt.axvspan(0, lower_start, alpha=0.5, color='red', label='Guard bands')
plt.axvspan(upper_end, FFT_Size, alpha=0.5, color='red')
plt.axvspan(lower_start, lower_end, alpha=0.2, color='lightgreen', label='User 1 subcarriers')
plt.axvline(x=256, color='red', alpha=1.0, linewidth=2, linestyle='--', label='DC null')
plt.axvspan(upper_start, upper_end, alpha=0.5, color='magenta', label='User 2 subcarriers')
plt.legend()

plt.subplot(2, 1, 2)
plt.stem(np.angle(fft_input[0, :]), linefmt='g-', markerfmt='go', basefmt=' ')
plt.xlabel('Subcarrier Index')
plt.ylabel('Phase (radians)')
plt.title('Phase of Subcarriers')
plt.grid(True, alpha=0.3)
# Lower guard band
plt.axvspan(0, lower_start, alpha=0.5, color='red')
# Upper guard band
plt.axvspan(upper_end, FFT_Size, alpha=0.5, color='red')
# User 1 subcarriers
plt.axvspan(lower_start, lower_end, alpha=0.2, color='lightgreen')
# DC null
plt.axvline(x=256, color='red', alpha=1.0, linewidth=2, linestyle='--', label='DC null')
# User 2 subcarriers
plt.axvspan(upper_start, upper_end, alpha=0.5, color='magenta')
plt.legend()
plt.tight_layout()
plt.savefig("OFDMA-Step-3-Frequency-Domain-Subcarrier-Allocation.png")
plt.show()

# ================== STAGE 4: IFFT & Cyclic Prefix Addition ==================
ofdm_time_domain = np.fft.ifft(fft_input, n=FFT_Size, axis=1) 

# ================== PLOT 4a: Time Domain OFDM Signal (Before CP) - First Symbol ==================
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(np.real(ofdm_time_domain[0, :]), 'b-', label='Real (I)')
plt.plot(np.imag(ofdm_time_domain[0, :]), 'r-', label='Imaginary (Q)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Stage 4a: Time Domain OFDM Signal After IFFT (First OFDM Symbol - Contains Both Users, No CP)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.abs(ofdm_time_domain[0, :]), 'g-')
plt.xlabel('Sample Index')
plt.ylabel('Magnitude')
plt.title('Magnitude of Time Domain Signal - First OFDM Symbol')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("OFDMA-Step-4a-Time-Domain-OFDM-Signal-After-IFFT-First-Symbol.png")
plt.show()

# ================== PLOT 4b: Time Domain OFDM Signal (Before CP) - Second Symbol ==================
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(np.real(ofdm_time_domain[1, :]), 'b-', label='Real (I)')
plt.plot(np.imag(ofdm_time_domain[1, :]), 'r-', label='Imaginary (Q)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Stage 4b: Time Domain OFDM Signal After IFFT (Second OFDM Symbol - Contains Both Users, No CP)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.abs(ofdm_time_domain[1, :]), 'g-')
plt.xlabel('Sample Index')
plt.ylabel('Magnitude')
plt.title('Magnitude of Time Domain Signal - Second OFDM Symbol')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("OFDMA-Step-4b-Time-Domain-OFDM-Signal-After-IFFT-Second-Symbol.png")
plt.show()

ofdm_with_cp = np.hstack((ofdm_time_domain[:, -cp_length_samples:], ofdm_time_domain))

# ================== PLOT 5a: Time Domain OFDM Signal (With CP) - First Symbol ==================
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(np.real(ofdm_with_cp[0, :]), 'b-', label='Real (I)')
plt.plot(np.imag(ofdm_with_cp[0, :]), 'r-', label='Imaginary (Q)')
plt.axvspan(0, cp_length_samples, alpha=0.3, color='yellow', label='Cyclic Prefix')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title(f'Stage 5a: Time Domain OFDM Signal With Cyclic Prefix (First OFDM Symbol - Contains Both Users, CP={cp_length_samples} samples)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.abs(ofdm_with_cp[0, :]), 'g-')
plt.axvspan(0, cp_length_samples, alpha=0.3, color='yellow')
plt.xlabel('Sample Index')
plt.ylabel('Magnitude')
plt.title('Magnitude of Time Domain Signal with CP - First OFDM Symbol')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("OFDMA-Step-5a-Time-Domain-OFDM-Signal-With-CP-First-Symbol.png")
plt.show()

# ================== PLOT 5b: Time Domain OFDM Signal (With CP) - Second Symbol ==================
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(np.real(ofdm_with_cp[1, :]), 'b-', label='Real (I)')
plt.plot(np.imag(ofdm_with_cp[1, :]), 'r-', label='Imaginary (Q)')
plt.axvspan(0, cp_length_samples, alpha=0.3, color='yellow', label='Cyclic Prefix')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title(f'Stage 5b: Time Domain OFDM Signal With Cyclic Prefix (Second OFDM Symbol - Contains Both Users, CP={cp_length_samples} samples)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.abs(ofdm_with_cp[1, :]), 'g-')
plt.axvspan(0, cp_length_samples, alpha=0.3, color='yellow')
plt.xlabel('Sample Index')
plt.ylabel('Magnitude')
plt.title('Magnitude of Time Domain Signal with CP - Second OFDM Symbol')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("OFDMA-Step-5b-Time-Domain-OFDM-Signal-With-CP-Second-Symbol.png")
plt.show()

# ================== STAGE 5: Transmission through AWGN Channel ==================
BER_ofdm_user1 = np.zeros(len(snr))
BER_ofdm_user2 = np.zeros(len(snr))

for snr_db in snr:
    tx_waveform = ofdm_with_cp.reshape(-1)
    signal_power = np.mean(np.abs(tx_waveform)**2)

    SNR_linear = 10**(snr_db/10)
    noise_power_per_complex_sample = signal_power / SNR_linear
    noise_variance = np.sqrt(noise_power_per_complex_sample/2)

    noise = noise_variance * (np.random.randn(*ofdm_with_cp.shape) + 1j*np.random.randn(*ofdm_with_cp.shape))

    received_signal = ofdm_with_cp + noise

    if snr_db == 10:
        # ================== PLOT 5c: Received Signal (Time Domain with Noise) - First Symbol ==================
        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        plt.plot(np.real(received_signal[0, :]), 'b-', label='Real (I)')
        plt.plot(np.imag(received_signal[0, :]), 'r-', label='Imaginary (Q)')
        plt.axvspan(0, cp_length_samples, alpha=0.3, color='yellow', label='Cyclic Prefix')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title(f'Stage 5c: Received Signal (First OFDM Symbol - Both Users, Time Domain with AWGN, SNR={snr_db} dB)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(np.abs(received_signal[0, :]), 'g-')
        plt.axvspan(0, cp_length_samples, alpha=0.3, color='yellow')
        plt.xlabel('Sample Index')
        plt.ylabel('Magnitude')
        plt.title('Magnitude of Received Signal - First OFDM Symbol')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"OFDMA-Step-5c-Received-Signal-Time-Domain-First-Symbol-SNR{snr_db}dB.png")
        plt.show()

        # ================== PLOT 5d: Received Signal (Time Domain with Noise) - Second Symbol ==================
        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        plt.plot(np.real(received_signal[1, :]), 'b-', label='Real (I)')
        plt.plot(np.imag(received_signal[1, :]), 'r-', label='Imaginary (Q)')
        plt.axvspan(0, cp_length_samples, alpha=0.3, color='yellow', label='Cyclic Prefix')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title(f'Stage 5d: Received Signal (Second OFDM Symbol - Both Users, Time Domain with AWGN, SNR={snr_db} dB)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(np.abs(received_signal[1, :]), 'g-')
        plt.axvspan(0, cp_length_samples, alpha=0.3, color='yellow')
        plt.xlabel('Sample Index')
        plt.ylabel('Magnitude')
        plt.title('Magnitude of Received Signal - Second OFDM Symbol')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"OFDMA-Step-5d-Received-Signal-Time-Domain-Second-Symbol-SNR{snr_db}dB.png")
        plt.show()

    # ================== STAGE 6: Receiver Processing ==================
    received_signal_no_cp = received_signal[:, cp_length_samples:]

    received_ofdm_symbols = np.fft.fft(received_signal_no_cp, n=FFT_Size, axis=1)

    received_lower = received_ofdm_symbols[:, lower_start:lower_end]
    received_upper = received_ofdm_symbols[:, upper_start:upper_end]

    # ================== STAGE 7: Demodulation & BER Calculation ==================
    const_points = np.array([constellation_I[s] + 1j*constellation_Q[s] for s in range(4)])

    received_lower_flat = received_lower.flatten()
    received_upper_flat = received_upper.flatten()
    
    distances_user1 = np.abs(received_lower_flat[:, None] - const_points[None, :])
    demod_symbols_user1 = np.argmin(distances_user1, axis=1)

    distances_user2 = np.abs(received_upper_flat[:, None] - const_points[None, :])
    demod_symbols_user2 = np.argmin(distances_user2, axis=1)

    bit_pairs = {
        0: [0, 0],
        1: [0, 1],
        2: [1, 0],
        3: [1, 1]
    }

    demod_bits_user1 = np.array([b for s in demod_symbols_user1 for b in bit_pairs[s]])
    demod_bits_user1 = demod_bits_user1[:num_bits]
    
    demod_bits_user2 = np.array([b for s in demod_symbols_user2 for b in bit_pairs[s]])
    demod_bits_user2 = demod_bits_user2[:num_bits]
    
    num_bit_errors_user1 = np.sum(user1_bits != demod_bits_user1)
    ber_user1 = num_bit_errors_user1 / num_bits
    BER_ofdm_user1[snr_db] = ber_user1

    num_bit_errors_user2 = np.sum(user2_bits != demod_bits_user2)
    ber_user2 = num_bit_errors_user2 / num_bits
    BER_ofdm_user2[snr_db] = ber_user2
    
    print(f"Bit Error Rate (BER) User 1: {ber_user1:.6e} for SNR = {snr_db} dB")
    print(f"Bit Error Rate (BER) User 2: {ber_user2:.6e} for SNR = {snr_db} dB")

    if snr_db == 10:
        tx_constellation_user1 = u1_qpsk_modulated[:1000]
        tx_constellation_user2 = u2_qpsk_modulated[:1000]
        tx_symbols_user1 = u1_symbols[:1000]
        tx_symbols_user2 = u2_symbols[:1000]
        rx_constellation_user1 = received_lower.flatten()[:1000]
        rx_constellation_user2 = received_upper.flatten()[:1000]

        # ================== PLOT 6: Received Signal in Frequency Domain ==================
        plt.figure(figsize=(14, 5))
        plt.subplot(2, 1, 1)
        plt.stem(np.abs(received_ofdm_symbols[0, :]), linefmt='b-', markerfmt='bo', basefmt=' ')
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Magnitude')
        plt.title(f'Stage 6: Received Signal in Frequency Domain (After FFT, SNR={snr_db} dB)')
        plt.grid(True, alpha=0.3)
        # Lower guard band
        plt.axvspan(0, lower_start, alpha=0.5, color='red', label='Guard bands')
        # Upper guard band
        plt.axvspan(upper_end, FFT_Size, alpha=0.5, color='red')
        # User 1 subcarriers
        plt.axvspan(lower_start, lower_end, alpha=0.2, color='lightgreen', label='User 1 subcarriers')
        # DC null
        plt.axvline(x=256, color='red', alpha=1.0, linewidth=2, linestyle='--', label='DC null')
        # User 2 subcarriers
        plt.axvspan(upper_start, upper_end, alpha=0.5, color='magenta', label='User 2 subcarriers')
        plt.legend(loc='upper left')
        
        plt.subplot(2, 1, 2)
        plt.stem(np.angle(received_ofdm_symbols[0, :]), linefmt='g-', markerfmt='go', basefmt=' ')
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Phase (radians)')
        plt.title('Phase of Received Subcarriers')
        plt.grid(True, alpha=0.3)
        # Lower guard band
        plt.axvspan(0, lower_start, alpha=0.5, color='red')
        # Upper guard band
        plt.axvspan(upper_end, FFT_Size, alpha=0.5, color='red')
        # User 1 subcarriers
        plt.axvspan(lower_start, lower_end, alpha=0.2, color='lightgreen')
        # DC null
        plt.axvline(x=256, color='red', alpha=1.0, linewidth=2, linestyle='--', label='DC null')
        # User 2 subcarriers
        plt.axvspan(upper_start, upper_end, alpha=0.5, color='magenta')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig("OFDMA-Step-6-Frequency-Domain-Received-Signal.png")
        plt.show()
        
        # ================== PLOT 7: Constellation Comparison User 1 vs User 2 ==================
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        
        max_lim = 6
        plt.fill([0, max_lim, max_lim, 0], [0, 0, max_lim, max_lim], alpha=0.1, color=symbol_colors[0])
        plt.fill([-max_lim, 0, 0, -max_lim], [0, 0, max_lim, max_lim], alpha=0.1, color=symbol_colors[1])
        plt.fill([-max_lim, 0, 0, -max_lim], [-max_lim, -max_lim, 0, 0], alpha=0.1, color=symbol_colors[3])
        plt.fill([0, max_lim, max_lim, 0], [-max_lim, -max_lim, 0, 0], alpha=0.1, color=symbol_colors[2])
        
        for sym in range(4):
            sym_indices = np.where(tx_symbols_user1 == sym)[0]
            sym_I = [rx_constellation_user1[i].real for i in sym_indices if i < len(rx_constellation_user1)]
            sym_Q = [rx_constellation_user1[i].imag for i in sym_indices if i < len(rx_constellation_user1)]
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
        plt.title(f'User 1 Received (SNR={snr_db} dB)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.legend(loc='upper right', framealpha=0.9, fontsize=10)

        plt.subplot(1, 2, 2)
        
        plt.fill([0, max_lim, max_lim, 0], [0, 0, max_lim, max_lim], alpha=0.1, color=symbol_colors[0])
        plt.fill([-max_lim, 0, 0, -max_lim], [0, 0, max_lim, max_lim], alpha=0.1, color=symbol_colors[1])
        plt.fill([-max_lim, 0, 0, -max_lim], [-max_lim, -max_lim, 0, 0], alpha=0.1, color=symbol_colors[3])
        plt.fill([0, max_lim, max_lim, 0], [-max_lim, -max_lim, 0, 0], alpha=0.1, color=symbol_colors[2])
        
        for sym in range(4):
            sym_indices = np.where(tx_symbols_user2 == sym)[0]
            sym_I = [rx_constellation_user2[i].real for i in sym_indices if i < len(rx_constellation_user2)]
            sym_Q = [rx_constellation_user2[i].imag for i in sym_indices if i < len(rx_constellation_user2)]
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
        plt.title(f'User 2 Received (SNR={snr_db} dB)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        plt.suptitle('Stage 7: User 1 vs User 2 Received Constellations', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("OFDMA-Step-7-Tx-Vs-Rx-Constellation-Diagrams-User1-vs-User2.png")
        plt.show()


# ================== PLOT 8: BER Performance (Simulated Only) ==================
plt.figure(figsize=(10, 6))
plt.plot(snr, BER_ofdm_user1, 'o-', linewidth=2, markersize=8, label='Simulated BER User 1')
plt.plot(snr, BER_ofdm_user2, 'ko-', linewidth=2, markersize=8, label='Simulated BER User 2')
plt.yscale('log')
plt.ylim([1e-6, 1])
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Bit Error Rate (BER)', fontsize=12)
plt.title('Stage 8: OFDM QPSK BER vs SNR Performance for User 1 vs User 2', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("OFDMA-Step-8-OFDM-BER-Just-Simulated.png")
plt.show()

# ================== PLOT 9: BER Performance with Theoretical Comparison ==================
plt.figure(figsize=(10, 6))
plt.plot(snr, BER_ofdm_user1, 'o-', linewidth=2, markersize=8, label='Simulated BER User 1')
plt.plot(snr, BER_ofdm_user2, 'ko-', linewidth=2, markersize=8, label='Simulated BER User 2')
plt.plot(snr, 0.5 * erfc(np.sqrt(10**(np.array(snr)/10))), 'r--', linewidth=2, label='Theoretical BER (QPSK AWGN)')
plt.yscale('log')
plt.ylim([1e-6, 1])
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Bit Error Rate (BER)', fontsize=12)
plt.title('Stage 9: OFDM QPSK BER Performance - Simulated vs Theoretical', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("OFDMA-Step-9-OFDM-BER-Sim-Theoretical-User1-vs-User2.png")
plt.show()