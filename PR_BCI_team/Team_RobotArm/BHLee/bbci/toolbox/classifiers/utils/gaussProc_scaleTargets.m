function [Ytrain, targetsMean, targetsStd] = gaussProc_scaleTargets(Ytrain, opt)
% gaussProc_scaleTargets - Helper function for GP models: Scale target values data if required
%
% Synopsis:
%   [Ytrain, targetsMean, targetsStd] = gaussProc_scaleTargets(Ytrain, opt)
%   
% Make a range check for the input. If features are poorly scaled, we can
% expect trouble with optimization

% If targetsMean and targetsStd are already passed in the options, use that
% values, otherwise compute from the targets
if isfield(opt, 'targetsMean') & ~isempty(opt.targetsMean),
  targetsMean = opt.targetsMean;
else
  targetsMean = mean(Ytrain);
end
if isfield(opt, 'targetsStd') & ~isempty(opt.targetsStd),
  targetsStd = opt.targetsStd;
else
  targetsStd = std(Ytrain);
end
if opt.scaletargets,
  Ytrain = (Ytrain-targetsMean)./targetsStd;
  if opt.verbosity>=1,
    fprintf('Option ''scaletargets'': Re-scaling target values to variance 1.\n');
  end
elseif opt.verbosity>=1,
  if abs(targetsMean)>1,
    fprintf(['Target values do not match the GP model assumption of a\n' ...
             'zero mean, unit std function (mean = %g, std = %g)\n' ...
             'This may cause trouble in training. Scale the targets\n' ...
             'appropriately, or use the option ''scaletargets''.\n'], ...
            targetsMean, targetsStd);
  end
end
