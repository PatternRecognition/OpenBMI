function [tp, fp, threshold, err] = val_tpfpRates(label, out, varargin)
% val_tpfpRates - True positive (TP) rates and false positive (FP) rates
%
% Synopsis:
%   [tp,fp] = val_tpfpRates(label,out)
%   [tp,fp,threshold,err] = val_tpfpRates(label,out,'Property',Value,...)
%   
% Arguments:
%  label: [2 N] matrix. Class membership matrix
%  out: [1 N nShuffles] matrix. Classifier output from xvalidation for
%      all N examples, nShuffles repetitions of the xvalidation.
%   
% Returns:
%  tp: [1 m] vector. True positive rate for the chosen false positive
%      rate(s). See comments below on the definitions.
%  fp: [1 m] vector. False positive rate for the chosen true positive
%      rate(s). See comments below on the definitions.
%  threshold: [1 m] vector. Classifier threshold used to obtain the TP/FP
%      rates
%  err: [1 m] vector. Error rates for the chosen FP or TP rates.
%   
% Properties:
%  tprate: [1 m] vector. True positive rates. Return argument fp are the
%      minimum false positive rates that are achieved when the TP rates
%      are required to be larger than the values given here. Default: []
%  fprate: [1 m] vector. False positive rates. Return argument tp are the
%      best true positive rates that are achieved with the FP rates not
%      exceeding the values given here. 
%      Default: [.01 .02 .05 .1 .2 .5 .9 .95 .98 .99]
%  detect_class: 1 or 2. The class to be detected, see
%      val_rocCurve. Default: 1
%  Furthermore, all valid properties of val_rocCurve are recognized. All
%  options are also passed to val_rocCurve to compute TP/FP rates.
%   
% Description:
%   This routine does a 'manual' evaluation of the ROC curves. Its
%   purpose is to check classifier performance when a particular true
%   positive (TP) rate or false positive (FP) rate is fixed by, for
%   example, application constraints. For chosen fixed TP or FP rates,
%   the other quantity (TP with given FP, or vice versa) is returned.
%
%   The following definitions of TP and FP rates are used:
%     TPrate = (true Positives) / (true Positives + false Negatives)
%            = p (classified as Positive | belongs to class Positive )
%            = Sensitivity = hit rate
%     FPrate = (false Positives) / (true Negatives + false Positives)
%            = p (classified as Positive | belongs to class Negative )
%            = 1 - Specificity = false alarm rate
%
%   Mind that sometimes the term FP rate is also defined to be
%     p (belongs to class Negative | classified as Positive )
%
% Examples:
%   
%   [tp,fp] = val_tpfpRates(label,out,'fprate', [.05 .1])
%   tp = 
%       0.81 0.92
%   fp = 
%       0.048 0.97
%   When fixing FP rates, return argument fp are the actually achieved FP
%   rates. The values are always lower than those passed in the fprate option.
%   
% See also: val_rocCurve
% 

% Author(s), Copyright: Anton Schwaighofer, May 2005
% $Id: val_tpfpRates.m,v 1.3 2005/05/17 12:13:12 neuro_toolbox Exp $

error(nargchk(2, inf, nargin));
opt= propertylist2struct(varargin{:});
[opt, isdefault] = ...
    set_defaults(opt, ...
                 'detect_class', 1, ...
                 'tprate', [], ...
                 'fprate', [0.01, 0.02, 0.05, 0.1, 0.2 0.5, 0.9, 0.95, 0.98, 0.99]);
% If tprate is specified, the default fprate is obsolete
if isdefault.fprate & ~isdefault.tprate;
  opt.fprate = [];
end
if ~xor(isempty(opt.tprate), isempty(opt.fprate)),
  error('You need to specify either ''tprate'' or ''fprate''');
end

% Compute the full ROC curve
[allTP, dummy1, allFP, dummy2, allThresh] = val_rocCurve(label, out, opt);

% To plot the steps in the ROC curve correctly, the retuned TP and FP
% vectors contain duplicated elements. Eg we might get
% tp = [... 0.6 0.6 0.8 ... ]
% fp = [... 0.3 0.5 0.5 ... ]
% In case we need to extract a value for fp=0.5, be pessimistic and
% return the 0.6 instead of the 0.8

if ~isempty(opt.tprate),
  if any(opt.tprate<0) | any(opt.tprate>1),
    error('Values for option ''tprate'' must be in the interval [0,1]');
  end
  % TP rates are given
  z = zeros(size(opt.tprate));
  tp = z;
  fp = z;
  threshold = z;
  for j = 1:length(tp),
    % Get the FP rate if we require to have at least this particular TP
    % rate
    exactMatch = find(allTP==opt.tprate(j));
    if length(exactMatch)>1,
      % Be pessimistic and return the lowest possible FP rate
      [dummy, ind] = min(allFP(exactMatch));
      ind = exactMatch(ind);
    else
      % The normal case, where we do not hit the steps exactly
      ind = find(allTP>=opt.tprate(j));
      ind = ind(1);
    end
    tp(j) = allTP(ind);
    fp(j) = allFP(ind);
    threshold(j) = allThresh(ind);
  end
else
  if any(opt.fprate<0) | any(opt.fprate>1),
    error('Values for option ''fprate'' must be in the interval [0,1]');
  end
  % FP rates are given
  z = zeros(size(opt.fprate));
  tp = z;
  fp = z;
  threshold = z;
  for j = 1:length(fp),
    % Get the best TP rate if we require to have no more than this
    % particular FP rate
    % Check whether we hit the steps exactly:
    exactMatch = find(allFP==opt.fprate(j));
    if length(exactMatch)>1,
      % Be pessimistic and return the lowest possible TP rate
      [dummy, ind] = min(allTP(exactMatch));
      ind = exactMatch(ind);
    else
      % The normal case, where we do not hit the steps exactly
      ind = find(allFP<=opt.fprate(j));
      ind = ind(end);
    end
    tp(j) = allTP(ind);
    fp(j) = allFP(ind);
    threshold(j) = allThresh(ind);
  end
end
N = sum(label,2);
if opt.detect_class==2,
  N = flipud(N);
end
% Re-compute the error rates for these values: At TP=1 and FP=1, we get all
% negatives wrongly classified and should obtain an error rate of
% N(2)/sum(N)
err = ((1-tp)*N(1)+fp*N(2))/sum(N);
