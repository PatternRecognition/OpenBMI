function cl= train_nusvm(x,y,nu,sigma)
%TRAIN_NUSVM trains a nu-svm for classication only with gaussian kernel.
%
% usage:
%   cl= train_nusvm(x,y,nu)
%
% inputs:
%   x       training data
%   y       training labels
%   nu      regularization constant
%   sigma   kernel width, thus the kernel is 
%                  k(xi,xj) = exp(-(xi-xj)'*(xi-xj)/sigma)
%
% outputs:
%   cl      the trained classifier
%
% sth * 25nov2004

start_cplex;

% change y to +1 and -1
y = 2*y(1,:)-1;

% calculate the kernel matrix
m = size(x,2);
xx = sum(x.*x,1);
K = exp(-(repmat(xx,[m 1]) + repmat(xx',[1 m]) - 2*x'*x)/sigma);
clear xx

% the quadratic programming problem, 
%
%     min  0.5*alpha'*diag(y)*K*diag(y)*alpha
%     s.t. 0<=alpha<=1/m
%          alpha'*y = 0
%          sum(alpha) >= nu
%
% for the following variables see help quadprog
H = diag(y)*K*diag(y);
f = zeros(m,1);
A   = -ones(1,m);  b   = -nu;
Aeq = y;           beq = 0;
lb  = zeros(m,1);  ub  = ones(m,1)/m;

switch 2
 case 1  % matlab optimization toolbox -- slow
  opts = optimset('Display','off','LargeScale','off');
  alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],opts);
 case 2  % cplex -- fast
  neq = 1;
  verb = 0;  % verbosity
  alpha = qp_solve(LPENV,sparse(H),f,sparse([Aeq;A]),[beq;b],lb,ub,neq,verb);
end

% calculate b and rho
pluss = find((y'==+1) & (0 < alpha) & (alpha < 1/m));
minus = find((y'==-1) & (0 < alpha) & (alpha < 1/m));
len = min(length(pluss),length(minus));
pluss = pluss(1:len);
minus = minus(1:len);
S = [pluss, minus];
b = -0.5*sum(K([pluss,minus],:)*diag(y)*alpha)/len;
rho = 0.5*(sum(K(pluss,:),1)-sum(K(minus,:),1)*diag(y)*alpha)/len;


% store the classifier
cl.sigma = sigma;
cl.x = x;
cl.y = y;
cl.alpha = alpha;
cl.b = b;
cl.nu = nu;
cl.rho = rho;
