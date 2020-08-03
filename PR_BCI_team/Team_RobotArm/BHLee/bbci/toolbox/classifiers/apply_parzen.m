function out= apply_parzen(C,x)
%APPLY_PARZEN applies a Parzen window approach for classication.
%
% usage:
%   out= apply_parzen(C,x)
%
% inputs:
%   C       a parzen classifier as created by train_parzen.m
%   x       test data
%
% outputs:
%   out     the estimated labels
%
% sth * 25nov2004

nte = size(x,2);
ntr = size(C.x,2);

% calculate all the distances between training and testing set
D = repmat(sum(x.*x,1),[ntr 1]) + repmat(sum((C.x).*(C.x),1)',[1 nte]) ...
    - 2 * (C.x)'*x;
D = exp(-D/(C.sigma));

% apply labels
D = D .* repmat((2*C.y(1,:)-1)',[1 nte]);
out = -sum(D,1);