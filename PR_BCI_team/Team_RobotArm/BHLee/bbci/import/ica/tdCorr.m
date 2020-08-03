function C= tdCorr(x, tau)
%C= tdCorr(x, tau);
%
% Symmetrisized time delayed correlations (assumes Ex'=0, signals as rows)
% as used by TDSEP (Andreas Ziehe, 1998)
 
[N, T]= size(x);
nTaus= length(tau);
 
C= zeros(N, N, nTaus);
for t= 1:nTaus,
  C0= x(:,1:T-tau(t))*x(:,1+tau(t):T)' / (T-tau(t)-1);
  C(:,:,t)= (C0+C0')/2;
end
