function CFY= train_RLSR(xTr, yTr, C)
%CFY= train_RLSR(xTr, yTr, C)
%
% regularized least squares regression to the labels, i.e.,
%   min[w,b]  1/2.w'w + 1/2.C/K.sum((w'x_k+b-y_k)^2)
%
% input: the constant C between 0 and inf

if size(yTr,1)==2 yTr = [-1 1]*yTr; end
[N,K]= size(xTr);
if ~exist('C') | isempty(C)
  C=1;
end

%determine matrix Q and vector c such that the derivative of the goal
%function (see above) regarding vector v=[w;b] is v'Q+c
Q= zeros(N,N);
for k= 1:K,
  Q= Q + xTr(:,k)*xTr(:,k)';
end
ck= C/K;
cks= ck*sum(xTr,2);
Q= [ck*Q+eye(N) cks; cks' C];
c= -2*ck*[xTr*yTr'; sum(yTr)];

%calculate zero of the derivative, i.e., minimum of the goal function
opt= -pinv(Q)*c;

CFY.w= opt(1:N);
CFY.b= opt(end);
