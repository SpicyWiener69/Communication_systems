% % LSequalizer_complex.m
% clear all; close all;
% 
% % Complex channel
% b = [0.5+0.1j, 1+0.2j, -0.6+0.3j];
% 
% % Complex QPSK source
% m = 1000;
% s = (sign(randn(1,m)) + 1j*sign(randn(1,m))) / sqrt(2);
% 
% % Channel output (add noise optionally)
% r = filter(b, 1, s);
% % r = r + (randn(1,m) + 1j*randn(1,m)) * sigma;  % add AWGN
% 
% n = 15;
% delta = 8;
% p = length(r) - delta;
% 
% % Build R and S (same structure, now complex-valued)
% R = toeplitz(r(n+1:p), r(n+1:-1:1));
% S = s(n+1-delta:p-delta).';
% 
% % KEY CHANGE: use ' (conjugate transpose) not .'
% f = inv(R'*R)*R'*S
% 
% Jmin = real(S'*S - S'*R*inv(R'*R)*R'*S)
% 
% y = filter(f, 1, r);
% 
% % QPSK decision
% dec = sign(real(y)) + 1j*sign(imag(y));
% dec = dec / sqrt(2);
% 
% err = 0.5*sum(abs(real(dec(delta+1:m)) - real(s(1:m-delta))) + ...
%              abs(imag(dec(delta+1:m)) - imag(s(1:m-delta))))

b = [0.5+0.1j, 1+0.2j, -0.6+0.3j];
m = 1000;
s = (sign(randn(1,m)) + 1j*sign(randn(1,m))) / sqrt(2);

[f, Jmin, err] = LSequalizer_complex(b, s, 15, 8, true);




function [f, Jmin, err] = LSequalizer_complex(b, s, n, delta, do_plot)
% LSequalizer_complex  Least-Squares linear equalizer for complex signals.
%
% INPUTS:
%   b        - channel coefficients (complex row vector)
%   s        - transmitted symbols, QPSK (complex row vector, length m)
%   n        - equalizer order (number of taps = n+1)
%   delta    - equalizer delay in samples (delta <= n*length(b))
%   do_plot  - boolean flag, true to show plots
%
% OUTPUTS:
%   f        - equalizer tap coefficients (complex column vector, length n+1)
%   Jmin     - minimum achievable MSE for this f and delta
%   err      - symbol error count after equalization
%

    if delta > n * length(b)
        warning('delta > n*length(b)');
    end

    m = length(s);

    % ------------------------------------------------------------------ %
    %  Channel output  (add noise line below if needed)
    % ------------------------------------------------------------------ %
    r = filter(b, 1, s);
    % sigma = 0.01;
    % r = r + (randn(1,m) + 1j*randn(1,m)) * sigma;

    % ------------------------------------------------------------------ %
    %  Build Toeplitz matrix R and desired vector S
    % ------------------------------------------------------------------ %
    p = length(r) - delta;
    R = toeplitz(r(n+1:p), r(n+1:-1:1));   % convolution matrix
    S = s(n+1-delta:p-delta).';             % desired output (training)

    % ------------------------------------------------------------------ %
    %  LS solution via normal equations  f = (R'R)^{-1} R'S
    % ------------------------------------------------------------------ %
    RtR = R' * R;
    f   = RtR \ (R' * S);                  % use \ instead of inv() for stability

    % ------------------------------------------------------------------ %
    %  Minimum MSE
    % ------------------------------------------------------------------ %
    Jmin = real(S' * S - S' * R * (RtR \ (R' * S)));

    % ------------------------------------------------------------------ %
    %  Apply equalizer and make decisions
    % ------------------------------------------------------------------ %
    y   = filter(f, 1, r);
    dec = (sign(real(y)) + 1j*sign(imag(y))) / sqrt(2);

    err = 0.5 * sum( ...
            abs(real(dec(delta+1:m)) - real(s(1:m-delta))) + ...
            abs(imag(dec(delta+1:m)) - imag(s(1:m-delta))) );

    fprintf('Jmin = %.4f\n', Jmin);
    fprintf('Symbol errors = %d / %d\n', err, m - delta);

    % ------------------------------------------------------------------ %
    %  Plots (only when requested)
    % ------------------------------------------------------------------ %
    if ~do_plot
        return;
    end

    figure('Name','LS Equalizer Results','NumberTitle','off');

    % --- Channel taps ---
    subplot(2,3,1);
    stem(0:length(b)-1, abs(b), 'filled');
    title('Channel |b|'); xlabel('Tap'); ylabel('Magnitude'); grid on;

    % --- Equalizer taps ---
    subplot(2,3,2);
    stem(0:length(f)-1, abs(f), 'filled');
    title('Equalizer |f|'); xlabel('Tap'); ylabel('Magnitude'); grid on;

    % --- Combined channel+equalizer ---
    c = conv(b, f.');
    subplot(2,3,3);
    stem(0:length(c)-1, abs(c), 'filled');
    title('Combined |b*f|'); xlabel('Tap'); ylabel('Magnitude'); grid on;

    % --- Received constellation ---
    subplot(2,3,4);
    plot(real(r), imag(r), '.', 'MarkerSize', 4);
    title('Received r'); xlabel('I'); ylabel('Q'); grid on; axis equal;

    % --- Equalized constellation ---
    subplot(2,3,5);
    plot(real(y), imag(y), '.', 'MarkerSize', 4);
    title('Equalized y'); xlabel('I'); ylabel('Q'); grid on; axis equal;

    % --- Decisions ---
    subplot(2,3,6);
    plot(real(dec), imag(dec), '.', 'MarkerSize', 4);
    title(sprintf('Decisions (err=%d)', err));
    xlabel('I'); ylabel('Q'); grid on; axis equal;

end