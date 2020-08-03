%function integt
%INTEGT	Unit test for the function INTEG.

%	O. Lemoine - March 1996.

N=128;

% Integration of a sinusoid on a period
x=(0:N);
y=sin(2*pi*x/N);
som=integ(y,x);
if abs(som)>sqrt(eps),
 error('integ test 1 failed');
end

% Integration of a constant
y=ones(1,N);
som=integ(y);
if abs(som-N+1)>sqrt(eps),
 error('integ test 2 failed');
end

% Integration of a ramp
y=(0:N); 
som=integ(y);
if abs(som-N*N/2)>sqrt(eps),
 error('integ test 3 failed');
end


N=113;

% Integration of a sinusoid on a period
x=(0:N);
y=sin(2*pi*x/N);
som=integ(y,x);
if abs(som)>sqrt(eps),
 error('integ test 4 failed');
end

% Integration of a constant
y=ones(1,N);
som=integ(y);
if abs(som-N+1)>sqrt(eps),
 error('integ test 5 failed');
end

% Integration of a ramp
y=(0:N); 
som=integ(y);
if abs(som-N*N/2)>sqrt(eps),
 error('integ test 6 failed');
end


