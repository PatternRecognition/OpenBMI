function loss = loss_negLogProb(label,pred,var,varargin)
% loss_negLogProb - Negative (Gaussian) log probability for regression
%
% Synopsis:
%   loss = loss_negLogProb(label,pred,var)
%   
% Arguments:
%  label: [1 N] vector of regression target values
%  pred: [1 N rep] matrix, predictive mean for each of the N test cases
%      in rep replications of crossvalidation
%  var: [1 N rep] matrix, predictive *variance* for each of the N test
%      cases in rep replications of crossvalidation
%   
% Returns:
%  loss: [1 N rep] matrix, loss incurred for each test case
%   
% Description:
%   Negative log probability is an error measure that takes both accuracy
%   of prediction and accuracy of error bars (predicted variance/standard
%   deviation) into account.
%   
%   
% Examples:
%   You are 0.5 off the target, but you predicted a 0.5 standard deviation:
%     loss_negLogProb(0, 0.5, 0.5^2)
%   But beware if you are 0.5 off the target, but predicted a 0.1
%   standard deviation: 
%     loss_negLogProb(0, 0.5, 0.1^2)
%   
% See also: xvalidation,normal_pdf
% 

% Author(s), Copyright: Anton Schwaighofer, Oct 2005
% $Id: loss_negLogProb.m,v 1.1 2007/02/16 15:07:18 neuro_toolbox Exp $

error(nargchk(3, 3, nargin));

rep = size(pred,3);
if rep>1,
  label = repmat(label, [1 1 rep]);
end
if any(var(:)<0),
  error('Some predictive variances are negative');
end
loss = 0.5*(log(2*pi*var) + ((pred-label).^2)./var);
