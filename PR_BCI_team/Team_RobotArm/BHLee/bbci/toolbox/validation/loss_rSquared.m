function R_Squared = loss_rSquared(targets, estimates)
%R_Squared = loss_rSquared(goal, out)
%
% IN  targets   - matrix of (multi-dimensional) goal values
%                 size [nDim nSamples]
%     estimates - matrix of outputs from regression
%                 size [nDim nSamples]
% OUT  R_Square - r-Square-value
%
% Function to compute the R² index proposed by d'Avella
% 
% Is actually no loss function, but a fitness fucntion. If you need a loss
% function, use loss_normMeanSquared.m!
%
% range from 1 to -inv,   1 means perfect regression,
%                         0 means chance level, meanSquaredError = var(targets)
%                        <0 worse than chance level
%
% 2012-04-18, janne.hahne@tu-berlin.de

nSamples = size(estimates, 2);
avgTargets = mean(targets, 2);


var_targets = mean(mean((targets - repmat(avgTargets,1,nSamples)).^2,1),2);

mse = mean(mean((estimates - targets).^2,1),2);

R_Squared = 1 - (mse / var_targets);

end