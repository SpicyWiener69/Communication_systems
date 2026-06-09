% qamser.m symbol error rate for QAM in additive noise
N0=10.^(0:-.2:-3);                % noise variances
N=10000;                          % # symbols to simulate

M=16;                              % # symbols in constellation
s=pam(N,sqrt(M),1)+j*pam(N,sqrt(M),1); % QAM symbols
s=s/sqrt(2);                      % normalize power

const=unique(real(s));            % normalized symbol values
allsers = zeros(size(N0));
for i=1:length(N0)                % loop over SNR
  n=sqrt(N0(i)/2)*(randn(1,N)+j*randn(1,N));
  r=s+n;                          % received signal+noise
  realerr=quantalph(real(r),const)==quantalph(real(s),const);
  imagerr=quantalph(imag(r),const)==quantalph(imag(s),const);
  SER=1-mean(realerr.*imagerr);   % determine SER by counting
  allsers(i)=SER;                 % # symbols w/ RE+IM correct
end

% decision=quantalph(real(r),const)+j*quantalph(imag(r),const);

pow(s)

tmp=3./(2*(M-1)*N0);
theory_SER=1-(1-(1-1/sqrt(M)).*erfc(sqrt(tmp))).^2;

figure(1)
semilogy(10*log10(1./N0),allsers,'bx',10*log10(1./N0),theory_SER,'r-')
axis([0 30 1e-5 1])
grid

figure(2)
plot(real(s),imag(s),'*')

figure(3)
decision=real(r)+j*imag(r);
plot(real(decision),imag(decision),'*') 


