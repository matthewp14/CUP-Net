function y = soft(x,T)

y = max(abs(hilbert(x)) - T, 0);
y = y./(y+T) .* x;

%y=x;
