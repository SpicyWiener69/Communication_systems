% LMSequalizerQAM.m 4-QAM LMS equalizer
clear all;
b=[0.5+0.2*j 1 -0.6-0.2*j];   % define complex channel
m=1000;                          % how many data points
s=pam(m,4,5)+j*pam(m,4,5);     % 4-QAM source of length m
r=filter(b,1,s);                 % output of channel
n=13; f=zeros(n,1);              % initialize equalizer at 0
mu=.001; delta=7;                % stepsize and delay delta
y=zeros(n,1);                    % place to store output
buf=[];
for i=n+1:m                      % iterate
    rr=r(i:-1:i-n+1);
    y(i)=rr*f;                   % output of equalizer
    e=s(i-delta)-y(i);           % calculate error term
    f=f+mu*e*rr';                % update equalizer coefficients
    buf=[buf f];
end

figure(1)
stem(real(b))
figure(2)
stem(real(f))
c=conv(b,f);
figure(3)
stem(real(c))
figure(4)
plot(r,'.')                    
figure(5)                        
y=filter(f,1,r);                 
plot(real(y(500:1000)),imag(y(500:1000)),'.')
figure(6)
plot(real(buf.'))