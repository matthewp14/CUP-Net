function y=MyTVpsi(x,th,tau,iter)

y=x-MyProjectionTV(x,tau,th*0.5,iter);

