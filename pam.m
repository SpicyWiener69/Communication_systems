function seq=pam(len,M,Var);
seq=(2*floor(M*rand(1,len))-M+1)*sqrt(3*Var/(M^2-1));
