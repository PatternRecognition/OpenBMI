function phi=if2phase(iflaw)
%IF2PHASE Generate the phase from the instantaneous frequency

%	O. Lemoine - February 1996.
% experimental
	 
N=length(iflaw);
phi=zeros(N,1);

%phi=2*pi*cumsum(iflaw);
for k=1:N,
 t=1:k;
 phi(k)=2*pi*integ(iflaw(t),t);
end