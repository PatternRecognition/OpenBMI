function [fv,w,la,c] = proc_csssp(fv,n,T,C);
%PROC_CSSSP TRAINS THE CSSSP ALGORITHM
%
% usage:
%    [fv,w,la,c] = proc_csssp(fv,n,T,C);
%
% input:
%     fv     feature vector
%     n      number of pattern for each class
%     T      length of FIR Filter -1  (default: 15)
%     C      sparsity constraint  (default:0)
%
% output:
%     fv     processed feature vector
%     w      spatial filter
%     la     csp value
%     c      temporal filter (for each class)
%
% Note: this is the implementation of NIPS 05 GD et al. 
%
% Guido Dornhege, 31/09/05
% $Id: proc_csssp.m,v 1.1 2005/08/31 08:33:29 neuro_cvs Exp $

if nargin<3 | isempty(T)
  T = 15;
end

if nargin<4 | isempty(C)
  C = 0;
end

[Sigma1,Sigma2] = calc_sigmas(fv,T);

par = optimset('MaxFunEvals',10000,'Display','off');

Sigma = Sigma1+Sigma2;

warning off
la = zeros(1,2*n);


a = fminunc('csssp_helper_func',zeros(T,1),par,1:T,Sigma1,Sigma,1,C);
b = fminunc('csssp_helper_func',zeros(T,1),par,1:T,Sigma2,Sigma,1,C);


warning on

w = zeros(size(Sigma1,1),2*n);

[dum,w(:,1:n),la(1:n)] = csssp_helper_func(a,1:T,Sigma1,Sigma,n,C);
[dum,w(:,n+1:2*n),la(n+1:2*n)] = csssp_helper_func(b,1:T,Sigma2,Sigma,n,C);

c = cat(2,a,b);

ind = find(abs(c)<0.0001*(1+repmat(sum(abs(c),1),[size(c,1),1])));
c(ind) = 0;

fv = proc_linearDerivation_filter(fv,w,c,T,n);


return;




function [Sigma1,Sigma2] = calc_sigmas(epo,T);

epo.x = epo.x-repmat(mean(epo.x,1),[size(epo.x,1),1,1]);

ind1 = find(epo.y(1,:)>0);
ind2 = find(epo.y(2,:)>0);
Sigma1 = zeros(size(epo.x,2),size(epo.x,2),T+1);
Sigma2 = zeros(size(epo.x,2),size(epo.x,2),T+1);

da = permute(epo.x,[2 1 3]);

for t = 0:T
  fprintf('\r%i/%i                      ',t,T+1);
  da3 = da(:,1:end-t,ind1);da3 = da3(:,:);
  da1 = da(:,t+1:end,ind1); da1 = da1(:,:);
  Sigma1(:,:,t+1) = (da1*da3'+da3*da1')/(size(da,2)-t-1)/(1+(t==0))/length(ind1);
  da3 = da(:,1:end-t,ind2);da3 = da3(:,:);
  da2 = da(:,t+1:end,ind2); da2 = da2(:,:);
  Sigma2(:,:,t+1) = (da3*da2'+da2*da3')/(size(da,2)-t-1)/(1+(t==0))/length(ind2);
end

fprintf('\n');




