Ts = 1/10000; 
time = 50;
t= Ts:Ts:time;

fc = 100;
phoff= -0.1;
r= cos(2*pi * fc * t + phoff);
fl = 100;
ff= [0 0.01 0.02 1];
fa = [1 0 0 0];
h = firpm(fl,ff,fa);
mu = 0.003;
f0 = 99.5;
fest = zeros(1,length(t));
fest(1) = f0 + 2;
z = zeros(1,fl+1);

for k = 1:length(t) -1 
    z = [(r(k) - cos(2*pi * fest(k) * t(k))) * sin(2 * pi * fest(k) * t(k)),z(1:fl)];
    update = fliplr(h) * z';
    fest(k+1) = fest(k) -2 * pi * k * Ts * mu * update;

end


plot(t,fest)