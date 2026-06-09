% LSequalizer.m find a LS equalizer f for the channel b
clear all;close all;
b=[0.5 1 -0.6];                        % define channel
m=5000; s=pam(m,2,1);            % binary source of length m
r=filter(b,1,s);    % output of channel
n = 15;f = zeros(n,1);
mu = 0.005;
delta = 8;
buf = [];

for i=n+1:m
    rr = r(i:-1:i-n+1).';
    e = s(i-delta) - rr.'*f;
    f = f+mu*e*rr;
    buf =[buf f];
end

figure(1)
stem(b)
axis([0 4 -1.5 1.5])

figure(2)
stem(f)

c=conv(b,f);
figure(3)
stem(c)

figure(4)
plot(r,'.');

figure(5)
y=filter(f,1,r);
plot(y,'.');


figure(6)
plot(real(buf.'))