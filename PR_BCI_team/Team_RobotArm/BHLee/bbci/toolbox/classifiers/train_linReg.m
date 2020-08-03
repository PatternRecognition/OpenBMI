function C= train_linReg(xTr, yTr,kappa)
% function C= train_linReg(xTr, yTr,kappa)
% 
% does linear regression with optional regularizer 0 <= kappa <= 1 (default = 0)
%
% works with standard apply-function apply_separatingHyperplane.m, so no
% apply-function needs to be defined
%
% 2012-02-18, janne.hahne@tu-berlin.de (code from felix)

if nargin<3,kappa=0;end

C.w = inv(xTr*xTr'+eye(size(xTr,1)).*kappa)*xTr*yTr';

%C.b = mean(yTr,2);
C.b = zeros(size(yTr,1),1);