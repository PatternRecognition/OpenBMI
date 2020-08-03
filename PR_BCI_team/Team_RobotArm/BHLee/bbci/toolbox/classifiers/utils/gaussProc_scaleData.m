function [Xtrain, featureMean, featureStd] = gaussProc_scaleData(Xtrain, opt)
% gaussProc_scaleData - Helper function for GP models: Scale training data if required
%
% Synopsis:
%   [Xtrain,featureMean,featureStd] = gaussProc_scaleData(Xtrain,opt)
%   

[dim N] = size(Xtrain);
% Make a range check for the input. If features are badly scaled, we can
% expect trouble with optimization
featureMean = mean(Xtrain')';
featureStd = std(Xtrain')';
if opt.scaledata,
  % Make sure the Std does not vanish:
  featureStd = featureStd+(featureStd<eps);
  % We should rescale the data before training: Scale by inverse standard
  % deviation. 
  Xtrain = diag(1./featureStd)*(Xtrain-repmat(featureMean, [1 N]));
  if opt.verbosity>=1,
    fprintf('Option ''scaledata'': Re-scaling all features to have unit variance.\n');
  end
elseif opt.verbosity>=1,
  if (any(featureStd<0.2) | any(featureStd>4)),
    fprintf(['Some features have particularly low or high variance.\n' ...
             '(minimum variance %g, maximum variance %g)\n' ...
             'This may cause trouble in training. Rescale the features to\n' ...
             'have approximately unit variance, choose suitable kernel inweights,\n'...
             'or use the option ''scaledata''.\n'], ...
            min(featureStd).^2, max(featureStd).^2);
  end
end
