function out= apply_nusvm(cl,x)
%APPLY_NUSVM applies the nu-SVM for classication.
%
% usage:
%   out= apply_nusvm(cl,x)
%
% inputs:
%   cl      a nusvm classifier as created by train_nusvm.m
%   x       test data
%
% outputs:
%   out     the estimated labels
%
% sth * 25nov2004

nte = size(x,2);
ntr = size(cl.x,2);

% calculate cross-kernel matrix between training and testing set
K = repmat(sum(x.*x,1),[ntr 1]) + repmat(sum((cl.x).*(cl.x),1)',[1 nte]) ...
    - 2 * (cl.x)'*x;
K = exp(-K/(cl.sigma));

out = -((cl.alpha)'*diag(cl.y)*K + (cl.b));

