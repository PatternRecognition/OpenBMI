% TRAIN_RegLLR - Train a Regularized LLR Classifier
%
% [C, info]=TRAIN_RegLLR(X, Y, fun, lambda, <opt>)
%  X : Data matrix (nDimension * nSamples)
%  Y : Labels      (nClasses   * nSamples)
% lambda : regularization constant
%  <opt>
%   .use_fminunc : use matlab minimization routine
%                 be aware that the license is limited
%                 (default 0)
%   .Display     : 'iter' for showing the iteration
%
% Multi-class cases are dealed as "one vs. the rest".
%
% Ryota Tomioka 2006
function [C, info]=train_RegLLR(X, Y, lambda, fun, varargin)

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'use_fminunc', 0,...
                        'Display', 'off',...
                        'Tol', 1e-9,...
                        'MaxIter', size(X,1)*1000);

[d, n] = size(X);
nCls = size(Y,1);

if ~exist('fun', 'var') | isempty(fun)
  fun = @objTrain_LogitLLR;
end

if n~=size(Y,2)
  error('sample size mismatch!');
end

iOK = find(any(Y));
if length(iOK)<n
  warning('TRAIN_REGLLR:REJECTION', 'Some samples were rejected before training the classy.');
  X=X(:,iOK);
  Y=Y(:,iOK);
  n = length(iOK);
end


C = repmat(struct('w', zeros(d,1), 'b', 0), [1, nCls]);

info = [];

i1 = 1+(nCls==2);

X = [X; ones(1, n)];

if opt.use_fminunc
  optfm = copy_struct(opt,'not','use_fminunc');
  
  if iscell(fun) & length(fun)>1
    optfm = optimset(optfm, 'GradObj','on','Hessian','on','HessMult', fun{2});
    fun=fun{1};
  else
    optfm = optimset(optfm, 'GradObj','on','Hessian','on');
  end
end

  

for i=i1:nCls
  x0 = zeros(d+1,1);
  y = 2*Y(i,:)-1;

  if opt.use_fminunc
    [x, fval, exitflag, output]=fminunc(fun, x0, optfm, X, y, lambda);
  else
    [x, fval, gg, exitflag]=newton(fun, x0, opt, X, y, lambda);
    output.firstorderopt = gg;
    output.algorithm = 'newton (within train_RegLLR)';
  end
    
  C(i).w = x(1:end-1);
  C(i).b = x(end);

  output.fval     = fval;
  output.exitflag = exitflag;

  if output.exitflag<=0
    warning('TRAIN_REGLLR:NONCONVERGENCE', 'fminunc did not converge.');
  end
  
  if isempty(info)
    info = output;
  else
    info = [info output];
  end
end

if nCls==2
  C=C(2);
end



function [x, f, gg, exitflag] = newton(fun, x0, opt, varargin)

x = x0;
gg = inf;

c = 0;

opt.Display = (isnumeric(opt.Display) & opt.Display~=0) | strcmp(opt.Display, 'iter');

if opt.Display
  fprintf('---------------------------------------------\n');
end

while gg>opt.Tol & c<opt.MaxIter
  [f, g, H] = feval(fun, x, varargin{:});
  x = x - inv(H)*g;
  gg = sum(abs(g));

  if opt.Display
    fprintf('[%d] f=%g\t%g\t%g\n', c, f, gg, det(H));
  end
  
  c = c + 1;
end
exitflag = c~=opt.MaxIter;

function [f, g, H] = objTrain_LogitLLR(x, X, Y, lambda)
% [f, g, H] = objTrain_LogitLLR(x, X, Y, lambda)
%
% l2-regularized LLR
%
% Ryota Tomioka 2006
[d, n] = size(X);

f = lambda/2*x'*x;
g = lambda*x;
H = lambda*eye(d);

for i=1:n
  expi = exp(-Y(i)*(x'*X(:,i)));
  p = expi/(1+expi);
  
  f = f + log(1+expi);
  g = g - Y(i)*X(:,i)*p;
  H = H + X(:,i)*X(:,i)'*p*(1-p);
end
