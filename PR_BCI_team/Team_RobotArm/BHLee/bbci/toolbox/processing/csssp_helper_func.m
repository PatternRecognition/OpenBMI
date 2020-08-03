function [res,w,la] = csssp_helper_func(aa,pos,Sigma1,Sigma2,n,C);
% CSSSP_HELPER_FUNC HELPER FUNCTION FOR PROC_CSSSP
% see documentation there.
%
% Guido Dornhege, 31/07/05
% $Id: csssp_helper_func.m,v 1.1 2005/08/31 08:33:41 neuro_cvs Exp $

if nargin<5 | isempty(n)
  n = 1;
end

if nargin<6 | isempty(C)
  C = 0;
end

[N,N,T] = size(Sigma1);


a = zeros(T,1);
a(1) = 1;
a(pos+1) = aa;


for i = 1:T
  b = a(1:T-i+1)'*a(i:T);

  Sigma1(:,:,i) = b*Sigma1(:,:,i);
  Sigma2(:,:,i) = b*Sigma2(:,:,i);
end

Sigma1 = sum(Sigma1,3);
Sigma2 = sum(Sigma2,3);

Sigma1 = (Sigma1+Sigma1');
Sigma2 = (Sigma2+Sigma2');

try
[w,la] = eigs(Sigma1,Sigma2,n,'LA',struct('disp',0));
la = diag(la);
res = la(1);
res = 1-res;

catch
  try
    res = 10+abs(min(eig(Sigma2)));
  catch
    res = 1000;
  end
  return;
end

if nargout>1
  w = w./repmat(sqrt(sum(w.*(Sigma2*w),1)),[size(w,1),1]);
end

if res<=0
  res = 10+abs(res);
  return;
end
  
res = res+C*sum(abs(aa))/length(aa);

