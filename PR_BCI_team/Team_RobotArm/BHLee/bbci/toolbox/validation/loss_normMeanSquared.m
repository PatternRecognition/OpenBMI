function loss = loss_normMeanSquared(targets, estimates)
%loss= loss_normMeanSquared(goal, out)
%
% IN  targets   - matrix of (multi-dimensional) goal values
%                 size [nDim nSamples]
%     estimates - matrix of outputs from regression
%                 size [nDim nSamples]
% OUT loss      - normalized mean square error (= 1 - R²)
%
% Function to compute mean square error, normalized by variance of the
% tragets
%
% range from 0 to inv,    0 means perfect regression,
%                         1 means chance level mean Squares = var(targets)
%                        >1 worse than chance level

loss = 1 - loss_rSquared(targets, estimates);