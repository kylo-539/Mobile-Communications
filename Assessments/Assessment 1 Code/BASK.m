clear
clc

% BASK - Modulation Scheme
sig = [1,0,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,1,0,0];

t_bit = 1; % Bit duration
fc = 3;    % Carrier Frequency
A1 = 5;    % Amplitude for bit = 1
A0 = 0;    % Amplitude for bit = 0
fs = 100;  % Sampling Frequency

t = 0:1/fs:t_bit - 1/fs; % Time vector for one bit

% Modulation Scheme
bask_signal = []; % Empty array to store the BASK Signal

for i = 1:length(sig)
    if sig(i) == 1
        a = A1 * cos(2*pi*fc * t);
    else
        a = A0 * cos(2*pi*fc * t);
    end
    bask_signal = [bask_signal a]; % Concatenate each bit's waveform
end

% Time vector for full signal
T = 0:1/fs:(t_bit*length(sig) - 1/fs);

%%
% Built in Amplitude Modulation
y = ammod(sig, fc, fs);


%%
%Demodulation Scheme
%received = bask_signal;    % Received signal
received = bask_signal + randn(size(bask_signal)); % Adding noise to the signal

samples_per_bit = length(t);
reference = cos(2*pi*fc*t);  % Reference carrier (same as transmitter)

demod = zeros(1, length(sig));  % Pre-allocate for speed

for i = 1:length(sig)
    % Extract the current bit segment
    segment = received((i-1)*samples_per_bit + 1 : i*samples_per_bit);

    % Multiply with reference carrier
    product = segment .* reference;

    % Average over bit period
    res = mean(product);

    % Threshold detection
    if res > 0
        demod(i) = 1;
    else
        demod(i) = 0;
    end
end

% Compare transmitted vs received bits
disp('Original bits:')
disp(sig)
disp('Demodulated bits:')
disp(demod)

%%
figure(1);
plot(T, bask_signal, 'k', T, received, 'g')
xlabel("Time (s)")
ylabel("Amplitude")
title("BASK Modulated Signal")
legend("Modulated Signal", "Received Signal")
grid on