function loss= loss_meanAbsolute(goal, out, varargin)
%loss= loss_meanAbsolute(goal, out)
%
% IN  goal - matrix of (multi-dimensional) goal values
%            size [nDim nSamples]
%     out  - matrix of outputs from regression
%                   
% OUT loss - vector of absolute differences
%
% SEE xvalidation

loss= sum(abs(out-goal), 1);
