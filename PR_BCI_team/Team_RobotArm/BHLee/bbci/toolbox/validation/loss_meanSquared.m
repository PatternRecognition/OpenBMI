function loss= loss_meanSquared(goal, out)
%loss= loss_meanSquared(goal, out)
%
% IN  goal - matrix of (multi-dimensional) goal values
%            size [nDim nSamples]
%     out  - matrix of outputs from regression
%                   
% OUT loss - vector of squared differences
%
% SEE xvalidation

loss= sum((out-goal).^2, 1);
