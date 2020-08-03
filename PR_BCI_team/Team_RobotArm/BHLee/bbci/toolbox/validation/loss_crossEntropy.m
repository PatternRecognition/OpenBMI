function loss= loss_crossEntropy(label, out, varargin)
% loss_crossEntropy - Cross entropy loss for binary classification
%
% Synopsis:
%   loss = loss_crossEntropy(label,out)
%   
% Arguments:
%  label: [2 N] matrix of class assignments
%  out: [1 N] vector of classifier outputs, must be in the range
%      0..1. Semantics: low classifier output must indicate class 1, thus
%      the classifier output is actually "probability for membership in
%      class 2"
%   
% Returns:
%  loss: Elementwise cross-entropy loss
%   
% Properties:
%  inf_loss: Scalar, maximum loss that can occur if predicting the wrong
%      class with absolute certainty (would be Inf otherwise). Default
%      value: Inf
%  max_loss: Scalar, maximum loss that can occur for "normal" predictions
%      that are not at the boundaries (probabilities of 0 or 1). Default
%      value: Inf
%   
% Description:
%   This is a loss function for probabilistic classification (classifier
%   outputs in the range [0..1], defined as
%   - log2(out)*class - log2(1-out)*(1-class)
%   with 0 / 1 class labels.
%   When the classifier output is 0 or 1, the loss can become Inf if the
%   wrong class is predicted. Use option inf_loss to threshold that.
%
%   A problem with this loss function is that it is asymmetric: A
%   prediction of 0.5 "costs" you a loss of 1, a perfectly correct
%   prediction can get you down to a loss of 0, but already a moderately
%   uncertain wrong prediction (e.g. prediction of 0.1 for a point in
%   class 1) costs you 3.3 bits loss. Option <max_loss> can be used to
%   specify a maximum loss that can occur for wrong predictions.
%   
% Examples:
%   loss_crossEntropy([1 0; 0 1], [0.1 0.2])
%    ans =
%            0.152       2.3219
%   first point is class 1, prediction 0.1, this gives low loss, but the
%   second point suffers high loss
%   loss_crossEntropy([1 0; 0 1], [0.1 0])
%    ans =
%            0.152       Inf
%   (loss get Inf when predicting the wrong thing with maximum certainty)
%   loss_crossEntropy([1 0; 0 1], [0.9 0], 'max_loss', 2)
%    ans =
%            2   Inf
%   (put a bound on loss for wrong preditions that are not maximally
%   certain)
%
% See also: xvalidation
% 

% Author(s), Copyright: Anton Schwaighofer, Aug 2007
% $Id: loss_crossEntropy.m,v 1.1 2007/08/28 11:52:58 neuro_toolbox Exp $

error(nargchk(2, inf,nargin))
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'inf_loss', Inf, 'max_loss', Inf);

if size(label,1)~=2,
  error('Cross-entropy only works for 2-class problems');
end
if any(out<0) || any(out>1),
  error('Cross-entropy only works with classifier outputs in the range [0..1]');
end
% Convert to 0 / 1 class labels
class = label2ind(label)-1;
% Maximum loss as default
loss = opt.inf_loss*ones(size(out));
out_zero = out==0;
out_one = (1-out)==0;
% No loss for correct predictions with maximum certainty
loss(out_zero & class==0) = 0;
loss(out_one & class==1) = 0;
others = ~out_zero & ~out_one;
others_loss = - class(others).*log2(out(others)) - (1-class(others)).*log2(1-out(others));
% cross entropy
loss(others) = min(opt.max_loss, others_loss);
