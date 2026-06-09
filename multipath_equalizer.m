% brute_force_dd.m t1
% idealized 16-APSK transmission system (refactored)
clear all; close all; clc;
global GAMMA CONSTELLATION GRAY_MAP

%% === CONSTELLATION SETUP ===
GAMMA = 3.15;
CONSTELLATION = build_constellation(GAMMA);
GRAY_MAP = [6 15 9 12 5 16 10 11 7 14 8 13 1 4 2 3];

figure(1)
plotconstellation(CONSTELLATION)
title('16APSK Constellation')


%% === TRAINING SEQUENCE ===
equalizer_order = 15;
equalizer_delta = 8;

N_train = equalizer_order + equalizer_delta + 10*(equalizer_order+1);   % = 183 minimum

% Random training symbols drawn from 16-APSK constellation
train_idx = randi(length(CONSTELLATION), 1, N_train);
s_train = CONSTELLATION(train_idx);


%% === MESSAGE SYMBOLS ===
str = '012345 hello world jzxv//[]-';

m = letters2apsk(str);
m_combined = [s_train m];
M = 100;    % upconvert times

[xi, xq] = upconvert(M, m_combined);

figure()
plotspec(xi + 1j*xq, 1/M)
title('Baseband 16APSK Signal')


%% === COMPLEX BASEBAND CHANNEL ===
% The problem requires a complex-valued channel.
% Therefore, apply the channel to the complex baseband signal.
r_channel = [0.5+0.1j, 1+0.2j, -0.6+0.3j];

x_bb = xi + 1j*xq;
x_bb_ch = filter(r_channel, 1, x_bb);

xi_ch = real(x_bb_ch);
xq_ch = imag(x_bb_ch);

figure()
plotspec(x_bb_ch, 1/M)
title('Complex Baseband Signal After Complex Channel')


%% === IQ MODULATION ===
T = 1;
fc = 20;

[r, t] = iq_modulate(xi_ch, xq_ch, T, M, fc);

figure()
plotspec(r, 1/M)
title('Transmitted 16APSK Passband Signal')


%% === RECEIVED PASSBAND SIGNAL ===
channel_sig = r;

figure()
plotspec(channel_sig, 1/M)
title('Received 16APSK Passband Signal')


%% === RECEIVER: COHERENT MIXING ===
[x2_i, x2_q] = coherent_mix(channel_sig, t, fc);

figure()
plotspec(x2_i + 1j*x2_q, 1/M)
title('Signal After Coherent Mixing')


%% === RECEIVER: LOW PASS FILTER ===
fl = 50;
[x3_i, x3_q, b_lpf] = lowpass_filter(x2_i, x2_q, fl);

figure()
freqz(b_lpf)
title('Low Pass Filter Frequency Response')


%% === RECEIVER: MATCHED FILTER ===
[y, yi, yq] = matched_filter(x3_i, x3_q, M);

figure()
plotspec(y, 1/M)
title('Matched Filter Output')


%% === EYE DIAGRAM ===
k0 = round(0.5*fl + M);
plot_eye_diagram(yi, k0, M);


%% === TIMING RECOVERY ===
N = length(m_combined);

[z, best_k, cost, k_start] = timing_recovery(y, N, M, k0, CONSTELLATION);

figure()
offset = (0:M-1)/M;
plot(offset, cost, 'LineWidth', 1.2);
grid on;
xlabel('Timing offset \tau');
ylabel('Cost function value');
title('Decision-Directed Timing Cost Function');

hold on;
best_ii = best_k - k_start + 1;
plot(offset(best_ii), cost(best_ii), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

figure()
plot(real(z), imag(z), 'b*');
hold on;
plot(real(CONSTELLATION), imag(CONSTELLATION), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
grid on; axis equal;
xlabel('I'); ylabel('Q');
title('16APSK Samples After Timing Recovery');
legend('Received samples', 'Ideal constellation');


%% === TRAINING DATA SPLIT ===
train_end = N_train;

z_train = z(1:train_end);
z_data  = z(train_end+1:end);

% Check alignment between received training block and known training symbols
[xc, lags] = simple_xcorr(z_train, s_train);
[~, peak] = max(abs(xc));

fprintf('Alignment lag = %d samples. Ideally this should be close to 0.\n', lags(peak));


%% === EQUALIZER DESIGN ===
[f, Jmin] = equalize_design(z_train, s_train, equalizer_order, equalizer_delta);

z_data_eq = equalize_apply(z_data, f, equalizer_delta);

% Reference message symbols after equalizer delay
m_ref = m(1:length(z_data_eq));
%m_ref = m(equalizer_delta+1 : equalizer_delta+length(z_data_eq));

%% === DECISION DEVICE ===
mprime = decision_device(z_data_eq, CONSTELLATION);

figure()
plot(real(mprime), imag(mprime), 'b*');
hold on;
plot(real(CONSTELLATION), imag(CONSTELLATION), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
grid on; axis equal;
xlabel('I'); ylabel('Q');
title('Detected 16APSK Symbols');
legend('Detected symbols', 'Ideal constellation');


%% === PERFORMANCE METRICS ===
[cvar, pererr, reconstructed_message] = evaluate_performance(mprime, z_data_eq, m_ref, CONSTELLATION);

fprintf('\n========== RESULT ==========\n');
fprintf('Cluster variance = %.6f\n', real(cvar));
fprintf('Symbol error rate = %.2f %%\n', pererr);
fprintf('Reconstructed message: %s\n', reconstructed_message);
fprintf('============================\n');


fprintf("length(m) = %d\n", length(m));
fprintf("length(z) = %d\n", length(z));
fprintf("length(z_data) = %d\n", length(z_data));
fprintf("length(z_data_eq) = %d\n", length(z_data_eq));
fprintf("length(mprime) = %d\n", length(mprime));


%% =========================================================
%% LOCAL FUNCTIONS
%% =========================================================

function CONSTELLATION = build_constellation(GAMMA)
    r1 = sqrt(16 / (4 + 12*GAMMA^2));
    r2 = GAMMA * r1;

    inner = r1 * exp(1j * (pi/4  + (0:3)  * pi/2));
    outer = r2 * exp(1j * (pi/12 + (0:11) * pi/6));

    CONSTELLATION = [inner, outer];
end


% function [xi, xq, m] = transmitter(str)
%     M  = 100;
%     m  = letters2apsk(str);
%     N  = length(m);
% 
%     mup_i = zeros(1, N*M + 1000);
%     mup_q = zeros(1, N*M + 1000);
% 
%     mup_i(1:M:N*M) = real(m);
%     mup_q(1:M:N*M) = imag(m);
% 
%     p  = hamming(M).';
%     xi = filter(p, 1, mup_i);
%     xq = filter(p, 1, mup_q);
% end


function [xi, xq] = upconvert(times, iq_message)
    N = length(iq_message);

    mup_i = zeros(1, N*times + 1000);
    mup_q = zeros(1, N*times + 1000);

    mup_i(1:times:N*times) = real(iq_message);
    mup_q(1:times:N*times) = imag(iq_message);

    p = hamming(times).';

    xi = filter(p, 1, mup_i);
    xq = filter(p, 1, mup_q);
end


function [r, t] = iq_modulate(xi, xq, T, M, fc)
    t = (1/M : 1/M : length(xi)/M) * T;

    c_i = cos(2*pi*fc*t);
    c_q = -sin(2*pi*fc*t);

    r = xi .* c_i + xq .* c_q;
end


function [x2_i, x2_q] = coherent_mix(r, t, fc)
    c2_i = cos(2*pi*fc*t);
    c2_q = -sin(2*pi*fc*t);

    x2_i = r .* c2_i;
    x2_q = r .* c2_q;
end


function [x3_i, x3_q, b] = lowpass_filter(x2_i, x2_q, fl)
    % Toolbox-free low-pass FIR filter using windowed sinc.
    % This avoids using firpm(), which may require toolbox.

    N = fl;
    n = 0:N;

    fc_norm = 0.12;      % normalized cutoff frequency
    mid = N/2;

    h = zeros(1, N+1);

    for k = 1:N+1
        if n(k) == mid
            h(k) = 2 * fc_norm;
        else
            h(k) = sin(2*pi*fc_norm*(n(k)-mid)) / (pi*(n(k)-mid));
        end
    end

    w = hamming(N+1).';
    b = h .* w;

    % Normalize DC gain
    b = b / sum(b);

    % Factor 2 compensates coherent demodulation loss
    x3_i = 2 * filter(b, 1, x2_i);
    x3_q = 2 * filter(b, 1, x2_q);
end


function [y, yi, yq] = matched_filter(x3_i, x3_q, M)
    p = hamming(M).';

    % Original code used pow(p), but pow() is not MATLAB built-in.
    % If pow(p) means average power, then pow(p)*M = sum(abs(p).^2).
    mf = fliplr(p) / sum(abs(p).^2);

    yi = filter(mf, 1, x3_i);
    yq = filter(mf, 1, x3_q);

    y = yi + 1j*yq;
end


function plot_eye_diagram(yi, k, M)
    ul = floor((length(yi) - k - 1) / (4*M));

    if ul <= 0
        warning('Not enough samples to plot eye diagram.');
        return;
    end

    figure()
    plot(reshape(yi(k : ul*4*M+k-1), 4*M, ul));
    axis([1 4*M -2 2]);
    grid on;
    title('Eye Diagram of I Branch');
    xlabel('Sample index');
    ylabel('Amplitude');
end


function [z, best_k, cost, k] = timing_recovery(y, N, M, k, CONSTELLATION)
    cost = zeros(1, M);

    for ii = 1:M
        kk = k + ii - 1;

        nmax = min(N, floor((length(y) - kk) / M) + 1);

        if nmax > 0
            ztemp = y(kk : M : kk + (nmax-1)*M);

            mtemp = decision_device(ztemp, CONSTELLATION);
            err = mtemp - ztemp;

            cost(ii) = sum(abs(err).^2) / length(err);
        else
            cost(ii) = NaN;
        end
    end

    [~, best_ii] = min(cost);

    best_k = k + best_ii - 1;

    nmax = min(N, floor((length(y) - best_k) / M) + 1);

    z = y(best_k : M : best_k + (nmax-1)*M);
end


function mhat = decision_device(z, CONSTELLATION)
    mhat = zeros(size(z));

    for n = 1:length(z)
        [~, idx] = min(abs(z(n) - CONSTELLATION));
        mhat(n) = CONSTELLATION(idx);
    end
end


function [cvar, pererr, reconstructed_message] = evaluate_performance(mprime, z, m, CONSTELLATION)
    cvar = (mprime - z) * (mprime - z)' / length(mprime);

    lmp = min(length(mprime), length(m));

    mprime = mprime(1:lmp);
    m_ref = m(1:lmp);

    idx_prime = zeros(1, lmp);
    idx_ref = zeros(1, lmp);

    for n = 1:lmp
        [~, idx_prime(n)] = min(abs(mprime(n) - CONSTELLATION));
        [~, idx_ref(n)] = min(abs(m_ref(n) - CONSTELLATION));
    end

    pererr = 100 * sum(idx_prime ~= idx_ref) / lmp;

    ltext = 2 * floor(length(mprime) / 2);

    reconstructed_message = apsk2letters(mprime(1:ltext));
end


function f = letters2apsk(str)
    global GRAY_MAP CONSTELLATION

    N = length(str);
    f = zeros(1, 2*N);

    for k = 0:N-1
        ascii_val = double(str(k+1));

        high_nibble = floor(ascii_val / 16);
        low_nibble = mod(ascii_val, 16);

        f(2*k+1) = CONSTELLATION(GRAY_MAP(high_nibble + 1));
        f(2*k+2) = CONSTELLATION(GRAY_MAP(low_nibble + 1));
    end
end


function str = apsk2letters(f)
    global GRAY_MAP CONSTELLATION

    inv_gray = zeros(1, 16);

    for i = 1:16
        inv_gray(GRAY_MAP(i)) = i - 1;
    end

    N = floor(length(f) / 2);
    str = char(zeros(1, N));

    for k = 0:N-1
        sym1 = f(2*k+1);
        sym2 = f(2*k+2);

        [~, idx1] = min(abs(sym1 - CONSTELLATION));
        [~, idx2] = min(abs(sym2 - CONSTELLATION));

        high_nibble = inv_gray(idx1);
        low_nibble = inv_gray(idx2);

        str(k+1) = char(high_nibble * 16 + low_nibble);
    end
end


function plotconstellation(CONSTELLATION)
    plot(real(CONSTELLATION), imag(CONSTELLATION), 'bo', ...
        'MarkerSize', 10, 'LineWidth', 2);

    grid on;
    axis equal;

    xline(0, 'k--', 'LineWidth', 1.2);
    yline(0, 'k--', 'LineWidth', 1.2);

    xlabel('I');
    ylabel('Q');
    title('16-Point Constellation');
end


function [f, Jmin] = equalize_design(z, m, n, delta)
    % equalize_design
    % Solve least-squares equalizer taps using training symbols.

    z = z(:).';
    m = m(:).';

    if length(z) ~= length(m)
        L = min(length(z), length(m));
        z = z(1:L);
        m = m(1:L);
    end

    if length(z) <= n + delta
        error('Training sequence is too short for this equalizer order and delay.');
    end

    p = length(z) - delta;

    R = toeplitz(z(n+1:p), z(n+1:-1:1));

    S = m(n+1-delta : p-delta).';

    RtR = R' * R;

    f = RtR \ (R' * S);

    Jmin = real(S'*S - S'*R*(RtR \ (R'*S)));

    fprintf('[equalize_design] n = %d, delta = %d, Jmin = %.6f\n', n, delta, Jmin);
end


function z_eq = equalize_apply(z, f, delta)
    z = z(:).';
    % Append zeros so all symbols flush through the equalizer
    z_padded = [z, zeros(1, delta)];
    y_full = filter(f, 1, z_padded);

    if length(y_full) <= delta
        error('equalize_apply: signal is too short for the selected equalizer delay.');
    end

    z_eq = y_full(delta+1:end);

    figure()
    plot(real(z_eq), imag(z_eq), '.', 'MarkerSize', 6);
    title('Equalized Received Symbols');
    xlabel('I');
    ylabel('Q');
    grid on;
    axis equal;
end


function [xc, lags] = simple_xcorr(x, y)
    % Toolbox-free cross-correlation.
    % This replaces xcorr().

    x = x(:).';
    y = y(:).';

    xc = conv(x, conj(fliplr(y)));

    lags = -(length(y)-1):(length(x)-1);
end
