function C = train_MSR(x, y, scaled)
%C = train_MSR(x, y, <scaled=0>)
%
% if class priors should be used set scale to 1.
% 
% see DUDA/HART MSE/Pseudoinverse

if ~exist('scaled','var'), scaled=0; end

if size(y,1)==2, y = [-1 1]*y; end

Y = [y', sparse(diag(y))*x'];
if scaled,
  b = (y==1)/sum(y==1) + (y==-1)/sum(y==-1);
else
  b= ones(size(y));
end

a = pinv(Y)*b';
C.w = a(2:end,1);
C.b = a(1,1);
