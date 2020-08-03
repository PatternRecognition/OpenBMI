function [fv, W, bias, out] = proc_LRR2(fv, C, varargin)
% proc_LRR2 - logistic regression with rank=2 approximation for binary classification problem
% f(V) = 1/2(-w1'*X*X'*w1+w2'*X*X'*w2)+bias
%
% [fv, W, bias, out] = proc_LRR2(fv, C, <opt>)
%
% Inputs:
%  fv    : epoched data. normal format or covarianced format (with proc_covariance)
%          WHITEN the data beforehand with proc_whiten.
%  C     : regularization constant
%          choose from e.g., exp(linspace(log(0.01),log(100),20))
%
%  <opt>
%  .W0        : initial value for W=[w1, w2]. default: random unit-orthogonal bases.
%  .weight    : 1xnSamples vector. sample weighting coefficients. default ones(1,nSamples)/n
%  .checkgrad : check analytic gradient.
%  .visualize : visualize the loss function.
%  .MaxIter   : number of iterations. default 1000.
%  .MaxFunEvals : defualt 1000*nChannels.
%  .TolFun      : default 1e-9.
%  .TolX      :   default 1e-9.
%
%
% Outputs:
%  fv         : the regression function value (the classifier output)
%  W=[w1, w2] : projection coefficients
%  bias       : the bias
%  out        : output of fminunc
%
% Example:
% =Train=
%  [epocv, Ww] = proc_whiten(proc_covariance(epoTr));
%  [fv, W, bias]  = proc_LRR2(epocv, 0.01);
%
% =Test =
%  fea=proc_variance(proc_linearDerivation(epoTe, Ww*W));
%  fea.x=0.5*[-1 1]*squeeze(fea.x)+bias;
%  err=1-mean((fea.x.*(fea.y(2,:)-.5))>0);
%
% See also:
%  proc_covariance, proc_whiten
%
% Feb 2007: Regularization constant C has the same weight
%           with a single sample, i.e., C -> C/n.
%
% Author: Ryota Tomioka. Jun, 2006.

[T,d,n] = size(fv.x);

%% Check binary assumption
ncls = 2;
if size(fv.y,1)~=ncls
  error('binary classification only!.');
end

%% index of valid samples
iValid=find(any(fv.y));


opt=propertylist2struct(varargin{:});
opt=set_defaults(opt, 'MaxIter', 1000, ...
                      'MaxFunEvals', d*1000, ...
                      'TolFun', 1e-9, ...
                      'TolX', 1e-9, ...
                      'display', 'off', ...
                      'W0', [], ...
                      'weight', ones(1,length(iValid))/length(iValid),...
                      'checkgrad', 0,...
                      'visualize', []);



if T==1
  d=sqrt(d);
  V=reshape(fv.x, [d, d, n]);
else
  V = zeros(d, d, n);
  for i=1:n
    V(:,:,i) = cov(fv.x(:,:,i));
  end
end

%% Convert class assignments into [-1 1] labels
Y = fv.y(2,:)*2-1;

%% check analytic gradient
if opt.checkgrad
  [r,x]=checkGrad(@objLRR2, 2*d+1, 10, 'symmetric', V(:,:,iValid), Y(iValid), C, opt.weight);
  figure, plot(1:2*d+1, r);
  keyboard;
end

%% visualization
if ~isempty(opt.visualize)
  W1=opt.visualize{1}.W;
  for i=1:length(opt.visualize)
    %% a small hack to get rid of the sign ambiguity
    Wtmp = opt.visualize{i}.W;
    Wtmp = Wtmp*diag(sign(diag(W1'*Wtmp)));
    xp(:,i)=[reshape(Wtmp, [2*d,1]); opt.visualize{i}.bias];
  end

  switch size(xp,2)
   case 2 % 1D visualization
    xl=-2:0.02:2;
    visualize1d(xl,V(:,:,iValid),Y(iValid),C,xp,opt);
    
   case 3 % 2D visualization
    xl=-2:0.2:2;
    visualize2d(xl,V(:,:,iValid),Y(iValid),C,xp,opt);
  end
end

if isempty(opt.W0)
  W0 = rand(d,2);
  [EV, ED]=eig(W0'*W0);
  W0=W0*EV*diag(1./sqrt(diag(ED)));
  opt.W0 = [W0(:,1); W0(:,2); 0];
elseif isequal(size(opt.W0), [d, 2])
  opt.W0 = [opt.W0(:,1); opt.W0(:,2); 0];
end  

optfmu = optimset('GradObj','on',...
                  'Hessian','on',...
                  'display',opt.display,...
                  'MaxIter', opt.MaxIter, ...
                  'MaxFunEvals', opt.MaxFunEvals, ...
                  'TolFun', opt.TolFun,...
                  'TolX', opt.TolX);

[W1, fval, exitflag, out] = fminunc(@objLRR2, opt.W0, optfmu, V(:,:,iValid), Y(iValid), C, opt.weight);

W = orthogonalize([W1(1:d), W1(d+1:2*d)]);
bias = W1(end);
x= [reshape(W,[2*d,1]); bias];
[f,g,H]=objLRR2(x,V(:,:,iValid),Y(iValid), C, opt.weight);

if f>fval+10*eps
  fprintf('orthogonalization icreases the loss...\n');
  W=[W1(1:d), W1(d+1:2*d)];
  out.fval = fval;
  [f,g,H]=objLRR2(W1,V(:,:,iValid),Y(iValid), C, opt.weight);
  out.g    = g;
  out.H    = H;
else
  out.fval = f;
  out.g    = g;
  out.H    = H;
end

out.exitflag = exitflag;

V = reshape(V, [d*d,n]);

fv.x = reshape(-0.5*W(:,1)*W(:,1)'+0.5*W(:,2)*W(:,2)', [1, d*d])*V+bias;


function W1 = orthogonalize(W)
% orthogonalization does not change the decision function
% but decreases the regulalization term
D=sqrtm(W'*W);
[EV, ED]=eig(D*diag([-1 1])*D');

W1 = W*inv(D)*EV*diag(sqrt(abs(diag(ED))));

% for debug
% W1'*W1
% rangeof(W*diag([-1 1])*W'-W1*diag([-1 1])*W1')

function [f, g, H] = objLRR2(x, V, Y, C, weight)
ncls = 2;
d = (size(x,1)-1)/ncls;
n = length(Y);

w1 = x(1:d);
w2 = x(d+1:2*d);
b  = x(end);

f = 0.5*C/n*x'*x;
g = C/n*x;


Hww1 = zeros(d,d);
Hww2 = zeros(d,d);
Hw1w2 = zeros(d,d);
Hw1b = zeros(d,1);
Hw2b = zeros(d,1);
Hbb  = 0;

for i=1:n
  expi = exp(-Y(i)*(-0.5*w1'*V(:,:,i)*w1+0.5*w2'*V(:,:,i)*w2+b));
  p = 1/(1+expi);
  
  Vw1 = V(:,:,i)*w1;
  Vw2 = V(:,:,i)*w2;
  
  
  f = f - log(p)*weight(i);
  gw1 =   Y(i)*Vw1*(1-p);
  gw2 = - Y(i)*Vw2*(1-p);
  gb  = - Y(i)*(1-p);
  g = g + [gw1; gw2; gb]*weight(i);
  
  Hww1  = Hww1  + (Vw1*Vw1'*p*(1-p) + Y(i)*V(:,:,i)*(1-p))*weight(i);
  Hww2  = Hww2  + (Vw2*Vw2'*p*(1-p) - Y(i)*V(:,:,i)*(1-p))*weight(i);
  Hw1w2 = Hw1w2  -Vw1*Vw2'*p*(1-p)*weight(i);
  Hw1b  = Hw1b   -Vw1*p*(1-p)*weight(i);
  Hw2b  = Hw2b  + Vw2*p*(1-p)*weight(i);
  Hbb   = Hbb   + p*(1-p)*weight(i);

end

H = C/n*eye(2*d+1) +...
    [Hww1,   Hw1w2, Hw1b;...
     Hw1w2', Hww2,  Hw2b;...
     Hw1b',  Hw2b', Hbb];


function visualize1d(X,V,Y,C,xp,opt)
d=size(V,1);
n=size(V,3);
F=zeros(size(X));
G=zeros(size(X));
loss=zeros(size(X));

dx = xp(:,2)-xp(:,1);

keyboard;
for i=1:length(X)
  x=xp(:,1)+dx*(X(i)+1)/2;
  [F(i),g]=objLRR2(x,V,Y, C, opt.weight);
  out = out1d(X(i),V,xp);
  % out = reshape(-0.5*x(1:d)*x(1:d)'+...
  %                0.5*x(d+1:2*d)*x(d+1:2*d)', [1, d*d])*...
  %        reshape(V,[d*d,n])+x(end);
  loss(i) = mean(out.*Y<0);
  G(i)=dx'*g;
end

figure, ax=plotyy(X, F, X, G);
ax(3)=axes('position',get(gca,'position'), 'YColor', [1 0 0], 'Color','none');
h=line(X, loss); set(h,'color',[1 0 0]);

axes(ax(1));
grid on;
hold on;
Ip=[find(X==-1), find(X==1)];
plot(X(Ip), F(Ip), 'ro', 'linewidth', 2);
set(ax(1),'Color','none')
for i=1:length(Ip)
  try
    text(X(Ip(i)), F(Ip(i)), opt.visualize{i}.desc, 'color', 'red');
  end
end

keyboard;

function visualize2d(X,V,Y,C,xp,opt)
d=size(V,1);
n=size(V,3);

[Xg,Yg]=meshgrid(X);
F=zeros(size(Xg));
G=zeros(size(Xg));
A=[1 -1 0; .5 .5 -1; 1 1 1]'; A=A*diag(1./sqrt(sum(A.^2)));
zp=A'*(eye(3)-ones(3)/3); zp=zp(1:2,:);
for i=1:size(zp,2)
  [Fp(i),g]=objLRR2(xp(:,i),V,Y, C, opt.weight);
  Gp(i)=norm(g);
end
for i=1:length(X)
  for j=1:length(X)
    x=xp*(ones(3,1)/3+A(:,1)*Xg(i,j)+A(:,2)*Yg(i,j));
    [F(i,j),g]=objLRR2(x,V,Y, C, opt.weight);
    G(i,j)=norm(g);
  end
end

figure, surf(Xg,Yg,F);
hold on;
plot3(zp(1,:), zp(2,:), Fp, 'ro', 'linewidth', 2);
for i=1:size(zp,2),
  try
    text(zp(1,i), zp(2,i), Fp(i)*1.1, opt.visualize{i}.desc, 'color', 'red');
  end
end

keyboard;

function cv=curv1d(V,xp)
d=size(V,1);
dx = xp(:,2)-xp(:,1);
cv=reshape(-.5*dx(1:d)*dx(1:d)'+...
           .5*dx(d+1:2*d)*dx(d+1:2*d)',[1,d*d])*...
   reshape(V,[d*d,size(V,3)]);

function out=out1d(X,V,xp)
d=size(V,1);
dx = xp(:,2)-xp(:,1);
for i=1:length(X)
  x=xp(:,1)+dx*(X(i)+1)/2;
  out(i,:)=reshape(-0.5*x(1:d)*x(1:d)'+...
                   0.5*x(d+1:2*d)*x(d+1:2*d)',[1,d*d])*...
           reshape(V,[d*d,size(V,3)])+x(end);
end