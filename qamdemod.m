% qamdemod.m: modulate and demodulate a complex-valued QAM signal
clear all;
N=10000; M=20; Ts=.0001; j=sqrt(-1);    % # symbols, oversampling factor
time=Ts*(N*M-1); t=0:Ts:time;           % sampling interval and time vectors
m=pam(N,2,1)+j*pam(N,2,1);             % signals of length N
ps=hamming(M);                           % pulse shape of width M
fc=1000; th=-1.0;                        % carrier freq. and phase

mup=zeros(1,N*M); mup(1:M:end)=m;      % oversample by integer length M
mp=filter(ps,1,mup);                    % convolve pulse shape with data
v=real(mp.*exp(j*(2*pi*fc*t+th)));      % complex carrier

f0=1000; ph=-1.0;                        % freq. and phase of demod
x=v.*exp(-j*(2*pi*f0*t+ph));           % demodulate v
l=50; f=[0,0.2,0.25,1]; a=[1 1 0 0];   % specify filter parameters
b=firpm(l,f,a);                          % design filter
s=2 * filter(b,1,x);                         % s=LPF{x}





%plots
figure(1),plotspec(mp,Ts)
figure(2),plotspec(v,Ts)
figure(3),plotspec(x,Ts)
figure(4),plotspec(s,Ts)
figure(5), plotspec(b,Ts)

figure(6),plot(real(s(1*M/2:M:end)),imag(s(1*M/2:M:end)),'*')
