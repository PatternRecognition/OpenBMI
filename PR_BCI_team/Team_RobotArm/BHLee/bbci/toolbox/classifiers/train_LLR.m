% TRAIN_LLR - Train a Linear Logistic Regression Classifier
% 
% [C, info]=TRAIN_LLR(X, Y)
%  X : Data matrix (nDimension * nSamples)
%  Y : Labels      (nClasses   * nSamples)
%
% Multi-class cases are dealed as "one vs. the rest".
%
% Ryota Tomioka 2006
function [C, info]=train_LLR(X, Y)

d=size(X,1);
iter = d*1000;

%[C, info]=train_RegLLR(X,Y, @objTrain_LogitLLR, 0, 'Display', 'off', 'MaxIter', iter);
[C, info]=train_RegLLR(X,Y, [], 0, 'Display', 'off', 'MaxIter', iter);
