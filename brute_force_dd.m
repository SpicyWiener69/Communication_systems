% brute_force_dd.m: idealized 16-APSK transmission system (refactored)
clear all; close all;
global GAMMA CONSTELLATION GRAY_MAP

%% === CONSTELLATION SETUP ===
GAMMA = 3.15;
CONSTELLATION = build_constellation(GAMMA);
GRAY_MAP = [6 15 9 12 5 16 10 11 7 14 8 13 1 4 2 3];

figure(1), plotconstellation(CONSTELLATION)
title('16APSK Constellation')


%% === TRANSMITTER ===

%% === TRAINING SEQUENCE ===
equalizer_order    = 15;
equalizer_delta   = 8;
N_train = equalizer_order + equalizer_delta + 10*(equalizer_order+1);   % = 183 minimum

% random QPSK-like symbols drawn from 16-APSK constellation
% uniform random → hits all 16 points, low autocorrelation
train_idx  = randi(length(CONSTELLATION), 1, N_train);
s_train    = CONSTELLATION(train_idx);   % known at both TX and RX

str = '012345 hello world jzxv//[]-=apsk';
%[xi, xq, m] = transmitter(str);

m  = letters2apsk(str);    
upconvert_times = 100;
m_combined = [s_train m];

[xi, xq] = upconvert(upconvert_times,m_combined);


figure(), plotspec(xi + 1j*xq, 1/100);
title('Baseband 16APSK Signal')


%% === IQ MODULATION ===
T = 1; M = 100; fc = 20;
[r, t] = iq_modulate(xi, xq, T, M, fc);

figure(), plotspec(r, 1/M)
title('tranmitted 16APSK Signal')


%% === CHANNEL EFFECTS: COHERENT MIXING ===
r_channel = [0.5+0.1j, 1+0.2j, -0.6+0.3j];
channel_sig = filter(r_channel, 1, r);

figure(), plotspec(channel_sig, 1/M)
title('recieved 16APSK Signal')

%% === RECEIVER: COHERENT MIXING ===
[x2_i, x2_q] = coherent_mix(channel_sig, t, fc);

figure(), plotspec(x2_i + 1j*x2_q, 1/M)
title('Signal After Coherent Mixing')


%% === RECEIVER: LOW PASS FILTER ===
fl = 50;
[x3_i, x3_q, b_lpf] = lowpass_filter(x2_i, x2_q, fl);

figure(), freqz(b_lpf)
title('Low Pass Filter Frequency Response')


%% === RECEIVER: MATCHED FILTER ===
M = 100;
[y, yi, yq] = matched_filter(x3_i, x3_q, M);

figure(), plotspec(y, 1/M)
title('Matched Filter Output')


%% === EYE DIAGRAM ===
fl = 50; k0 = 0.5*fl + M;
plot_eye_diagram(yi, k0, M);


%% === TIMING RECOVERY (DD) ===
N = length(m_combined);
[z, best_k, cost, k0] = timing_recovery(y, N, M, k0, CONSTELLATION);

figure()
offset = (0:M-1)/M;
plot(offset, cost); grid
xlabel('timing offset \tau'); ylabel('value of cost function')
title('Decision-Directed Timing Cost Function')
hold on
best_ii = best_k - k0 + 1;
plot(offset(best_ii), cost(best_ii), 'ro', 'MarkerSize', 10, 'LineWidth', 2)
hold off

figure()
plot(real(z), imag(z), 'b*'); hold on
plot(real(CONSTELLATION), imag(CONSTELLATION), 'ro', 'MarkerSize', 10, 'LineWidth', 2)
hold off; grid on; axis equal
xlabel('I'); ylabel('Q')
title('16APSK Samples After Timing Recovery')
legend('Received samples', 'Ideal constellation')

%%=== TRAINING DATA SPLIT ===
train_end   = N_train ;
z_train     = z(1 : train_end);          % T/M-spaced training block
z_data      = z(train_end+1 : end);      % T/M-spaced data block


% quick sanity check: correlation between z_train and s_train
% should peak at lag 0 if alignment is correct
[xc, lags] = xcorr(z_train, s_train);
[~, peak]  = max(abs(xc));
fprintf('alignment lag = %d samples (should be 0)\n', lags(peak));


%%===ADD EQUALIZER ==========
% equalizer_order    = 15;
% delta_eq = 8;
[f, Jmin] = equalize_design(z_train, s_train, equalizer_order, equalizer_delta);
z_data_eq      = equalize_apply(z_data, f, equalizer_delta);
m_ref     = m(1:length(z_data_eq));


%% === DECISION DEVICE ===
mprime = decision_device(z_data_eq, CONSTELLATION);

figure()
plot(real(mprime), imag(mprime), 'b*'); hold on
plot(real(CONSTELLATION), imag(CONSTELLATION), 'ro', 'MarkerSize', 10, 'LineWidth', 2)
hold off; grid on; axis equal
xlabel('I'); ylabel('Q')
title('Detected 16APSK Symbols')
legend('Detected symbols', 'Ideal constellation')


%% === PERFORMANCE METRICS ===
[cvar, pererr, reconstructed_message] = evaluate_performance(mprime, z_data_eq, m, CONSTELLATION)


%% =========================================================
%%  LOCAL FUNCTIONS
%% =========================================================

% ---------------------------------------------------------
% Build 16-APSK constellation from gamma
% ---------------------------------------------------------
function CONSTELLATION = build_constellation(GAMMA)
    r1 = sqrt(16 / (4 + 12*GAMMA^2));
    r2 = GAMMA * r1;
    inner = r1 * exp(1j * (pi/4  + (0:3)  * pi/2));
    outer = r2 * exp(1j * (pi/12 + (0:11) * pi/6));
    CONSTELLATION = [inner, outer];
end


% ---------------------------------------------------------
% Transmitter: encode string → upsample → pulse shape
% Returns upsampled I/Q arrays and the complex symbol sequence
% ---------------------------------------------------------
function [xi, xq, m] = transmitter(str)
    M  = 100;
    m  = letters2apsk(str);
    N  = length(m);

    mup_i = zeros(1, N*M + 1000);
    mup_q = zeros(1, N*M + 1000);
    mup_i(1:M:N*M) = real(m);
    mup_q(1:M:N*M) = imag(m);

    p  = hamming(M).';
    xi = filter(p, 1, mup_i);
    xq = filter(p, 1, mup_q);
end

function [xi,xq] = upconvert(times,iq_message)
  N = length(iq_message)

  mup_i = zeros(1, N*times + 1000);
  mup_q = zeros(1, N*times + 1000);
  mup_i(1:times:N*times) = real(iq_message);
  mup_q(1:times:N*times) = imag(iq_message);
  p  = hamming(times).';
  xi = filter(p, 1, mup_i);
  xq = filter(p, 1, mup_q);

end



% ---------------------------------------------------------
% IQ modulation: baseband → passband
% ---------------------------------------------------------
function [r, t] = iq_modulate(xi, xq, T, M, fc)
    t   = (1/M : 1/M : length(xi)/M) * T;
    c_i = cos(2*pi*fc*t);
    c_q = -sin(2*pi*fc*t);
    r   = xi.*c_i + xq.*c_q;
end


% ---------------------------------------------------------
% Coherent mixing: passband → baseband I/Q
% ---------------------------------------------------------
function [x2_i, x2_q] = coherent_mix(r, t, fc)
    c2_i = cos(2*pi*fc*t);
    c2_q = -sin(2*pi*fc*t);
    x2_i = r .* c2_i;
    x2_q = r .* c2_q;
end


% ---------------------------------------------------------
% Low-pass filter both I and Q branches
% Returns filtered signals and the filter coefficients
% ---------------------------------------------------------
function [x3_i, x3_q, b] = lowpass_filter(x2_i, x2_q, fl)
    fbe   = [0 0.1 0.2 1];
    damps = [1 1 0 0];
    b     = firpm(fl, fbe, damps);
    x3_i  = 2 * filter(b, 1, x2_i);
    x3_q  = 2 * filter(b, 1, x2_q);
end


% ---------------------------------------------------------
% Matched filter
% Returns complex output y and separate I/Q branches
% ---------------------------------------------------------
function [y, yi, yq] = matched_filter(x3_i, x3_q, M)
    p  = hamming(M).';
    mf = fliplr(p) / (pow(p) * M);
    yi = filter(mf, 1, x3_i);
    yq = filter(mf, 1, x3_q);
    y  = yi + 1j*yq;
end


% ---------------------------------------------------------
% Eye diagram plot (I branch)
% ---------------------------------------------------------
function plot_eye_diagram(yi, k, M)
    ul = floor((length(yi) - k - 1) / (4*M));
    figure(7)
    plot(reshape(yi(k : ul*4*M+k-1), 4*M, ul));
    axis([1 4*M -2 2]); grid
    title('Eye Diagram of I Branch')
    xlabel('Sample index'); ylabel('Amplitude')
end


% ---------------------------------------------------------
% Decision-directed timing recovery
% Sweeps M candidate offsets, returns downsampled z at best timing
% ---------------------------------------------------------
function [z, best_k, cost, k] = timing_recovery(y, N, M, k, CONSTELLATION)
    cost = zeros(1, M);

    for ii = 1:M
        kk   = k + ii - 1;
        nmax = min(N, floor((length(y) - kk) / M) + 1);

        if nmax > 0
            ztemp = y(kk : M : kk + (nmax-1)*M);
            mtemp = decision_device(ztemp, CONSTELLATION);
            err   = mtemp - ztemp;
            cost(ii) = sum(abs(err).^2) / length(err);
        else
            cost(ii) = NaN;
        end
    end

    [~, best_ii] = min(cost);
    best_k = k + best_ii - 1;

    nmax = min(N, floor((length(y) - best_k) / M) + 1);
    z    = y(best_k : M : best_k + (nmax-1)*M);
end


% ---------------------------------------------------------
% Decision device: nearest constellation point for each sample
% ---------------------------------------------------------
function mhat = decision_device(z, CONSTELLATION)
    mhat = zeros(size(z));
    for n = 1:length(z)
        [~, idx]  = min(abs(z(n) - CONSTELLATION));
        mhat(n)   = CONSTELLATION(idx);
    end
end


% ---------------------------------------------------------
% Performance metrics: cluster variance, symbol error %, decoded text
% ---------------------------------------------------------
function [cvar, pererr, reconstructed_message] = evaluate_performance(mprime, z, m, CONSTELLATION)
    cvar = (mprime - z) * (mprime - z)' / length(mprime);

    lmp      = min(length(mprime), length(m));
    m_ref    = m(1:lmp);
    idx_prime = zeros(1, lmp);
    idx_ref   = zeros(1, lmp);

    for n = 1:lmp
        [~, idx_prime(n)] = min(abs(mprime(n) - CONSTELLATION));
        [~, idx_ref(n)]   = min(abs(m_ref(n)  - CONSTELLATION));
    end

    pererr = 100 * sum(idx_prime ~= idx_ref) / lmp;

    ltext = 2 * floor(length(mprime) / 2);
    reconstructed_message = apsk2letters(mprime(1:ltext));
end


% ---------------------------------------------------------
% Encode ASCII string as 16-APSK complex symbol sequence
% ---------------------------------------------------------
function f = letters2apsk(str)
    global GRAY_MAP CONSTELLATION
    N = length(str);
    f = zeros(1, 2*N);

    for k = 0:N-1
        ascii_val   = double(str(k+1));
        high_nibble = floor(ascii_val / 16);
        low_nibble  = mod(ascii_val, 16);
        f(2*k+1)    = CONSTELLATION(GRAY_MAP(high_nibble + 1));
        f(2*k+2)    = CONSTELLATION(GRAY_MAP(low_nibble  + 1));
    end
end


% ---------------------------------------------------------
% Decode 16-APSK complex symbol sequence back to ASCII string
% ---------------------------------------------------------
function str = apsk2letters(f)
    global GRAY_MAP CONSTELLATION

    inv_gray = zeros(1, 16);
    for i = 1:16
        inv_gray(GRAY_MAP(i)) = i - 1;
    end

    N   = length(f) / 2;
    str = char(zeros(1, N));

    for k = 0:N-1
        sym1 = f(2*k+1);
        sym2 = f(2*k+2);

        [~, idx1]   = min(abs(sym1 - CONSTELLATION));
        [~, idx2]   = min(abs(sym2 - CONSTELLATION));
        high_nibble = inv_gray(idx1);
        low_nibble  = inv_gray(idx2);
        str(k+1)    = char(high_nibble * 16 + low_nibble);
    end
end


% ---------------------------------------------------------
% Plot constellation points
% ---------------------------------------------------------
function plotconstellation(CONSTELLATION)
    plot(real(CONSTELLATION), imag(CONSTELLATION), 'bo', 'MarkerSize', 10, 'LineWidth', 2)
    grid on; axis equal
    xline(0, 'k--', 'LineWidth', 1.2);
    yline(0, 'k--', 'LineWidth', 1.2);
    title('16-Point Constellation')
end



function [f, Jmin] = equalize_design(z, m, n, delta)
% equalize_design  Solve for LS equalizer taps using training symbols.
%
% Uses the Normal Equations:  f = (R'R)^{-1} R'S
% where R is the Toeplitz convolution matrix of z,
% and S is the desired output (training symbols delayed by delta).
%
% INPUTS:
%   z      - received symbol-rate samples, complex row vector, length N
%            output of timing_recovery()
%   m      - known TX symbols (training sequence), complex row vector
%            output of transmitter(); same m used throughout pipeline
%   n      - equalizer order; filter has n+1 taps
%   delta  - equalizer delay in samples; recommend delta ~ n/2
%            must satisfy delta <= n * channel_length
%
% OUTPUTS:
%   f      - equalizer tap coefficients, complex column vector, length n+1
%   Jmin   - minimum MSE for this f and delta (real scalar)

    % sanity check
    if delta > n * 10
        warning('equalize_design: delta may be too large relative to n.');
    end

    p = length(z) - delta;

    % --- build Toeplitz convolution matrix R from received signal ---
    % each row is a window of n+1 samples of z
    R = toeplitz(z(n+1:p), z(n+1:-1:1));

    % --- build desired output vector S (training, shifted by delta) ---
    S = m(n+1-delta : p-delta).';   % complex column vector

    % --- LS solution via normal equations ---
    % use \ for numerical stability instead of explicit inv()
    RtR = R' * R;
    f   = RtR \ (R' * S);

    % --- minimum achievable MSE ---
    Jmin = real(S'*S - S'*R*(RtR \ (R'*S)));

    fprintf('[equalize_design]  n=%d  delta=%d  Jmin=%.6f\n', n, delta, Jmin);
end


% ---------------------------------------------------------

function z_eq = equalize_apply(z, f, delta)
% equalize_apply  Apply pre-computed FIR equalizer to received samples.
%
% Filters z with tap vector f, then trims the leading delta samples
% to realign the output with the TX symbol timing.
%
% INPUTS:
%   z      - received symbol-rate samples, complex row vector
%            can be the training block or a new data block
%   f      - equalizer taps from equalize_design(), complex column vector
%   delta  - same delta used in equalize_design()
%
% OUTPUT:
%   z_eq   - equalized symbol-rate samples, complex row vector
%            length = length(z) - delta
%            feeds directly into decision_device()
%
% NOTE: the caller must trim the reference symbol vector m consistently:
%       m_ref = m(1 : length(z_eq))

    % apply FIR equalizer
    y_full = filter(f, 1, z);

    % trim leading delta samples to realign with TX symbols
    z_eq = y_full(delta+1 : end);
    
    figure()
    plot(real(z_eq), imag(z_eq), '.', 'MarkerSize', 4);
    title('Received r'); xlabel('I'); ylabel('Q'); grid on; axis equal;
end

