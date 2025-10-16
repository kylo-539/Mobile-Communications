# Digital Modulation Schemes Comparison

## Overview
This project implements and compares three fundamental digital modulation schemes: **BASK (Binary Amplitude Shift Keying)**, **QPSK (Quadrature Phase Shift Keying)**, and **8PSK (8-Phase Shift Keying)**. The implementation includes complete simulation, constellation analysis, and Bit Error Rate (BER) performance evaluation.

## 🎯 Features

- **Complete modulation/demodulation simulation** for three schemes
- **Constellation diagram generation** with decision regions
- **BER vs Eb/N0 performance analysis** with theoretical comparisons
- **AWGN channel modeling** with proper noise variance scaling
- **Correlation receiver implementation** with maximum likelihood detection
- **Comprehensive visualization** with publication-quality plots

## 📊 Modulation Schemes

### BASK (Binary Amplitude Shift Keying)
- **Data Rate**: 1 bit per symbol
- **Constellation**: Unipolar (0V for bit '0', 5V for bit '1')
- **Symbol Duration**: 1 second
- **Energy per Bit**: Eb = A₁²×Tbit/4
- **Decision Threshold**: A₁/4 = 1.25

### QPSK (Quadrature Phase Shift Keying)
- **Data Rate**: 2 bits per symbol
- **Constellation**: 4-point with Gray coding (45°, 135°, 225°, 315°)
- **Symbol Duration**: 2 seconds
- **Energy per Bit**: Eb = (A²×Tsymbol/2)/2
- **Detection**: Dual I/Q correlation with 4-point ML

### 8PSK (8-Phase Shift Keying)
- **Data Rate**: 3 bits per symbol
- **Constellation**: 8-point circular (22.5° spacing)
- **Symbol Duration**: 3 seconds
- **Energy per Bit**: Eb = (A²×Tsymbol/2)/3
- **Detection**: Dual I/Q correlation with 8-point ML

## 🔧 Requirements

```python
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
pandas>=1.3.0
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/kylo-539/Mobile-Communications.git
cd Mobile-Communications/Assessments
```

2. Install dependencies:
```bash
pip install numpy matplotlib scipy pandas
```

3. Run the simulation:
```bash
python CompleteCode.py
```

## 🔬 Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Bits Simulated** | 100,000 | Total bits for Monte Carlo simulation |
| **Carrier Frequency** | 3 Hz | RF carrier frequency |
| **Sampling Frequency** | 100 Hz | Digital sampling rate |
| **Signal Amplitude** | 5V | Peak signal amplitude |
| **Eb/N0 Range** | 0-15 dB | Energy per bit to noise ratio |
| **Random Seed** | 42 | For reproducible results |

## 📈 Key Results

### BER Performance (at 15 dB Eb/N0)
- **BASK**: ~10⁻⁴ BER
- **QPSK**: ~10⁻⁵ BER (Best performance)
- **8PSK**: ~10⁻⁴ BER

### Spectral Efficiency
- **BASK**: 1 bit/symbol
- **QPSK**: 2 bits/symbol (2× improvement)
- **8PSK**: 3 bits/symbol (3× improvement)

### Implementation Complexity
- **BASK**: Single correlation, threshold detection
- **QPSK**: Dual I/Q correlation, 4-point ML detection
- **8PSK**: Dual I/Q correlation, 8-point ML detection

## 🎨 Generated Visualizations

### 1. Constellation Diagrams
- **Transmitted**: Ideal constellation points
- **Received**: Noisy constellation with decision regions
- **Comparison**: Side-by-side visualization of all schemes

### 2. BER Performance Curves
- Simulated vs theoretical BER curves
- Waterfall characteristic curves
- Performance comparison across schemes

### 3. Decision Regions
- BASK: 1D threshold detection
- QPSK: 4-quadrant decision boundaries
- 8PSK: 8-sector pie-slice regions

## 📝 Technical Implementation

### Energy Calculations
```python
# BASK (Unipolar)
Eb_bask = (A1**2) * t_bit_bask / 4

# QPSK  
Eb_qpsk = (A**2 * t_symbol_qpsk / 2) / 2

# 8PSK
Eb_8psk = (A**2 * t_symbol_8psk / 2) / 3
```

### Noise Modeling
```python
# Convert Eb/N0 to linear scale
EbN0_linear = 10**(EbN0_dB / 10)

# Calculate noise parameters
N0 = Eb / EbN0_linear
noise_variance = N0 * fs / 2
noise_std = sqrt(noise_variance)
```

### Correlation Detection
```python
# I/Q correlation for PSK schemes
I_corr = sum(rx_segment * I_carrier) * 2 / fs
Q_corr = sum(rx_segment * Q_carrier) * 2 / fs

# Maximum Likelihood detection
min_distance = min(|I_corr - I_constellation|² + |Q_corr - Q_constellation|²)
```

## 🔍 Key Findings

1. **QPSK provides optimal balance** between spectral efficiency and BER performance
2. **8PSK suffers ~4-5 dB penalty** compared to QPSK due to reduced minimum distance
3. **BASK shows superior low-SNR performance** but poor spectral efficiency
4. **Gray coding absence in 8PSK** increases bit error propagation
5. **Simulation matches theory** well at high Eb/N0 values

## 📚 Theoretical Background

### BER Formulas
- **BASK (Unipolar)**: BER = 0.5 × erfc(√(Eb/2N0))
- **QPSK**: BER = 0.5 × erfc(√(Eb/N0))
- **8PSK**: BER ≈ (2/3) × erfc(√(3×Eb/N0×sin²(π/8)))

### Minimum Distance Analysis
- **QPSK**: dmin = 2√2×A
- **8PSK**: dmin = 2×A×sin(π/8) ≈ 0.76×A

## 🎓 Educational Value

This project demonstrates:
- Digital communication system design principles
- Monte Carlo simulation techniques
- Constellation diagram interpretation
- BER performance analysis methodology
- Trade-offs between spectral efficiency and error performance

## 📧 Contact

**Kyle Sheehy**  
Course: EEN1043 Wireless/Mobile Communications  
Institution: DCU

## 📄 License

This project is created for educational purposes as part of university coursework.

---

*Generated constellation diagrams and BER curves provide comprehensive analysis of digital modulation scheme performance characteristics.*
