% ideal_16apsk_timing.m: idealized 16-APSK transmission system
global GAMMA CONSTELLATION GRAY_MAP

%% === CONSTELLATION SETUP ===
GAMMA = 3.15;
r1 = sqrt(16 / (4 + 12*GAMMA^2));
r2 = GAMMA * r1;

inner = r1 * exp(1j * (pi/4  + (0:3)  * pi/2));
outer = r2 * exp(1j * (pi/12 + (0:11) * pi/6));

CONSTELLATION = [inner, outer];

GRAY_MAP = [6 15 9 12 ...
            5 16 10 11 7 14 8 13 1 4 2 3];

figure(1), plotconstellation(CONSTELLATION)
title('16APSK Constellation')


% TRANSMITTER
% encode text string as 16-APSK complex symbol sequence
% each character becomes two 16APSK symbols

str = '012345 hello world jzxv//[]-=';
m = letters2apsk(str); N = length(m);  % number of 16APSK symbols

% zero pad T-spaced symbol sequence to create upsampled
% T/M-spaced sequence of scaled T-spaced pulses
T = 1;
M = 100;                       % oversampling factor
fl = 50;                       % low-pass filter length

mup_i = zeros(1, N*M + 1000);  % in-phase upsample grid
mup_q = zeros(1, N*M + 1000);  % quadrature upsample grid
mup_i(1:M:N*M) = real(m);
mup_q(1:M:N*M) = imag(m);
p = hamming(M).';              % make it a row vector
xi = filter(p, 1, mup_i);
xq = filter(p, 1, mup_q);

figure(2), plotspec(xi + 1j*xq, 1/M)
title('Baseband 16APSK Signal')


% IQ modulation
t = (1/M:1/M:length(xi)/M) * T;    % T/M-spaced time vector
fc = 20;                           % carrier frequency
c_i = cos(2*pi*fc*t);              % in-phase carrier
c_q = -sin(2*pi*fc*t);             % quadrature carrier
r = xi.*c_i + xq.*c_q;             % passband transmitted signal

figure(3), plotspec(r, 1/M)
title('Passband 16APSK Signal')


% RECEIVER
c2_i = cos(2*pi*fc*t);             % synchronized cosine
c2_q = -sin(2*pi*fc*t);            % synchronized sine
x2_i = r.*c2_i;                    % demodulated I branch
x2_q = r.*c2_q;                    % demodulated Q branch

figure(4), plotspec(x2_i + 1j*x2_q, 1/M)
title('Signal After Coherent Mixing')


% LPF
fbe = [0 0.1 0.2 1];               % LPF frequency band edges
damps = [1 1 0 0];                 % desired amplitudes
b = firpm(fl, fbe, damps);         % create LPF impulse response

figure(5), freqz(b)
title('Low Pass Filter Frequency Response')

x3_i = 2*filter(b, 1, x2_i);
x3_q = 2*filter(b, 1, x2_q);


% matched filter
mf=fliplr(p)/(pow(p)*M);
yi = filter(mf, 1, x3_i);
yq = filter(mf, 1, x3_q);
y = yi + 1j*yq;

figure(6), plotspec(y, 1/M)
title('Matched Filter Output')


% eye diagram
k = 0.5*fl + M;                 % nominal first sampling point
ul = floor((length(yi)-k-1)/(4*M));
figure(7)
plot(reshape(yi(k:ul*4*M+k-1),4*M,ul));
axis([1 4*M -2 2])
grid
title('Eye Diagram of I Branch')
xlabel('Sample index')
ylabel('Amplitude')


% timing recovery (DD)
cost = zeros(1,M);
for ii = 1:M
    kk = k + ii - 1;                 % candidate sampling time
    nmax = min(N, floor((length(y)-kk)/M) + 1);
    if nmax > 0
        % sample the matched filter output using this timing offset
        ztemp = y(kk:M:kk+(nmax-1)*M);
        % decision device for this candidate offset
        mtemp = zeros(size(ztemp));
        
        for n = 1:length(ztemp)
            [~,idx] = min(abs(ztemp(n) - CONSTELLATION));
            mtemp(n) = CONSTELLATION(idx);
        end

        % decision-directed cost function
        % smaller cost means better timing
        err = mtemp - ztemp;
        cost(ii) = sum(abs(err).^2)/length(err);
    else
        cost(ii) = NaN;
    end
end

% find best timing offset
[~,best_ii] = min(cost);
best_k = k + best_ii - 1;

offset = (0:M-1)/M;

figure(8)
plot(offset,cost)
grid
xlabel('timing offset \tau')
ylabel('value of cost function')
title('Decision-Directed Timing Cost Function')

hold on
plot(offset(best_ii),cost(best_ii),'ro','MarkerSize',10,'LineWidth',2)
hold off


% downsample using recovered timing

nmax = min(N, floor((length(y)-best_k)/M) + 1);

z = y(best_k:M:best_k+(nmax-1)*M);      % soft decisions after timing recovery

figure(9)
plot(real(z),imag(z),'b*')
hold on
plot(real(CONSTELLATION),imag(CONSTELLATION),'ro','MarkerSize',10,'LineWidth',2)
hold off
grid on
axis equal
xlabel('I')
ylabel('Q')
title('16APSK Samples After Timing Recovery')
legend('Received samples','Ideal constellation')


% decision device
mprime = zeros(size(z));
for n = 1:length(z)
    [~,idx] = min(abs(z(n) - CONSTELLATION));
    mprime(n) = CONSTELLATION(idx);
end

figure(10)
plot(real(mprime),imag(mprime),'b*')
hold on
plot(real(CONSTELLATION),imag(CONSTELLATION),'ro','MarkerSize',10,'LineWidth',2)
hold off
grid on
axis equal
xlabel('I')
ylabel('Q')
title('Detected 16APSK Symbols')
legend('Detected symbols','Ideal constellation')


cvar = (mprime-z)*(mprime-z)'/length(mprime)    % cluster variance
lmp = min(length(mprime),length(m));            % symbol error percentage
m_ref = m(1:lmp);
idx_prime = zeros(1,lmp);
idx_ref = zeros(1,lmp);
for n = 1:lmp
    [~,idx_prime(n)] = min(abs(mprime(n) - CONSTELLATION));
    [~,idx_ref(n)]   = min(abs(m_ref(n)  - CONSTELLATION));
end
pererr = 100*sum(idx_prime ~= idx_ref)/lmp

% decode decision device output to text string
ltext = 2*floor(length(mprime)/2);
reconstructed_message = apsk2letters(mprime(1:ltext))


%% LOCAL FUNCTIONS
function f = letters2apsk(str)  
    global GRAY_MAP CONSTELLATION
    N = length(str);
    f = zeros(1, 2*N);            % complex output, 2 symbols for each character
    
    for k = 0:N-1
        ascii_val = double(str(k+1));  
    
        % Split 8 bits into two 4-bits (high, low)
        high_nibble = floor(ascii_val / 16);   % bits 7-4  
        low_nibble  = mod(ascii_val, 16);      % bits 3-0  (0..15)
    
        f(2*k+1) = CONSTELLATION(GRAY_MAP(high_nibble + 1));
        f(2*k+2) = CONSTELLATION(GRAY_MAP(low_nibble + 1));
    end
    
end 
    

function str = apsk2letters(f)
    global GRAY_MAP CONSTELLATION 

    inv_gray = zeros(1, 16);
    for i = 1:16
        inv_gray(GRAY_MAP(i)) = i - 1;   % GRAY_MAP(i) holds 1-based index
    end
    
    N   = length(f) / 2;          
    str = char(zeros(1, N));       % preallocate output 
    
    for k = 0:N-1
        sym1 = f(2*k+1);           % high nibble symbol
        sym2 = f(2*k+2);           % low nibble symbol
    
        % Minimum Euclidean distance decision for each symbol
        [~, idx1] = min(abs(sym1 - CONSTELLATION));  
        [~, idx2] = min(abs(sym2 - CONSTELLATION));
    
        % Recover nibbles via inverse Gray map
        high_nibble = inv_gray(idx1);   % 0..15
        low_nibble  = inv_gray(idx2);   % 0..15
    
        ascii   = high_nibble * 16 + low_nibble;
        str(k+1)    = char(ascii);
    end
end

function plotconstellation(CONSTELLATION)
    plot(real(CONSTELLATION), imag(CONSTELLATION), 'bo','MarkerSize',10,'LineWidth',2)
    
    grid on
    axis equal
    xline(0,'k--','LineWidth',1.2);   % Imaginary axis (vertical)
    yline(0,'k--','LineWidth',1.2);   % Real axis (horizontal)

    title('16-Point Constellation')


end