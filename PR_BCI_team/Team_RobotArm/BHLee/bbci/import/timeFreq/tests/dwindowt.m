%function dwindowt
%DWINDOWT Unit test for the function DWINDOW.

%	O. Lemoine - March 1996.

N=5000; 

% Rectangular window
h=window(N,'rect');
Dh1=[1;zeros(N-2,1);-1];
Dh2=dwindow(h);
if any(abs(Dh1-Dh2)>sqrt(eps)),
 error('dwindow test 1 failed');
end;

% Hanning window
h=window(N,'hanning');
Dh1=pi*sin(2*pi*(1:N)'/(N+1))/(N+1);
Dh2=dwindow(h);
if any(abs(Dh1(2:N-1)-Dh2(2:N-1))>sqrt(eps)),
 error('dwindow test 2 failed');
end;

% Bartlett window
h=window(N,'bartlett');
Dh1=[2*ones(N/2-1,1);1;-1;-2*ones(N/2-1,1)]/N;
Dh2=dwindow(h);
if any(abs(Dh1(2:N-1)-Dh2(2:N-1))>sqrt(eps)*10),
 error('dwindow test 3 failed');
end;

% Papoulis window
h=window(N,'papoulis');
Dh1=pi*cos(pi*(1:N)'/(N+1))/(N+1);
Dh2=dwindow(h);
if any(abs(Dh1(2:N-1)-Dh2(2:N-1))>sqrt(eps)*10),
 error('dwindow test 4 failed');
end;

% Gaussian window
K=0.02; h=window(N,'gauss',K);
t=linspace(-1,1,N)';
Dh1=4*log(K)*t.*exp(log(K)*t.^2)/N;
Dh2=dwindow(h);
if any(abs(Dh1(2:N-1)-Dh2(2:N-1))>sqrt(eps)*10),
 error('dwindow test 5 failed');
end;

N=4977; 

% Rectangular window
h=window(N,'rect');
Dh1=[1;zeros(N-2,1);-1];
Dh2=dwindow(h);
if any(abs(Dh1-Dh2)>sqrt(eps)),
 error('dwindow test 6 failed');
end;

% Hanning window
h=window(N,'hanning');
Dh1=pi*sin(2*pi*(1:N)'/(N+1))/(N+1);
Dh2=dwindow(h);
if any(abs(Dh1(2:N-1)-Dh2(2:N-1))>sqrt(eps)),
 error('dwindow test 7 failed');
end;

% Papoulis window
h=window(N,'papoulis');
Dh1=pi*cos(pi*(1:N)'/(N+1))/(N+1);
Dh2=dwindow(h);
if any(abs(Dh1(2:N-1)-Dh2(2:N-1))>sqrt(eps)*10),
 error('dwindow test 8 failed');
end;

% Gaussian window
K=0.02; h=window(N,'gauss',K);
t=linspace(-1,1,N)';
Dh1=4*log(K)*t.*exp(log(K)*t.^2)/N;
Dh2=dwindow(h);
if any(abs(Dh1(2:N-1)-Dh2(2:N-1))>sqrt(eps)*10),
 error('dwindow test 9 failed');
end;