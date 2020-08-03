function [Xtrain,opt,selected] = gaussProc_featureFilter(Xtrain,Ytrain,opt)
% gaussProc_featureFilter - Filter features that are uncorrelated with target value
%
% Synopsis:
%   [Xtrain,opt,selected] = gaussProc_featureFilter(Xtrain,Ytrain,opt)
%   
% Arguments:
%  Xtrain: [dim N] matrix of training points
%  Ytrain: [1 N] matrix, target values or class indicators
%  opt: GP option structure, required fields are 'featureFilter' (name of
%      the actual filter function) and 'featurethresh' (threshold for feature
%      p-values)
%   
% Returns:
%  Xtrain: [dim2 N] matrix with feature subset
%  opt: updated GP options structure, with option ''kernelindex'' updated
%      to the selected subset
%  selected: [dim 1] logical array, with ones indicating those features
%      that are retained, with nnz(selected)==dim2
%   
% Description:
%   This function returns a subset of those features of Xtrain that are
%   correlated with the target values. Correlation is measured by calling
%   filter functions specified in opt.featureFilter. If the value
%   returned by this function is smaller than a threshold specified by
%   opt.featurethresh, the corresponding feature is selected.
%
%   Filter functions must follow the syntax
%     func(fv, param1, param2, ...)
%   where fv is a struct array with fields 'x' ([dim N] training data)
%   and 'y', ([1 N] target values)
%   
% Examples:
%   opt.featureFilter = {'proc_p_permutationtest', 'perms', 2000};
%   opt.featurethresh = 0.05;
%   Xtrain = gaussProc_featureFilter(Xtrain, Ytrain, opt);
%     will return the subset of features that have a p-value <= 0.05 in a
%     permutation test with 2000 permutations.
%   
% See also: 
% 

% Author(s), Copyright: Anton Schwaighofer, Oct 2005
% $Id: gaussProc_featureFilter.m,v 1.1 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(3, 3, nargin));

opt = set_defaults(opt, 'featurefilter', {}, ...
                        'featurethresh', 0.05);
[dim N] = size(Xtrain);
if isempty(opt.featurefilter),
  selected = logical(ones([dim 1]));
  return;
end

% filter of the form 'string'
if ischar(opt.featurefilter) | isa(opt.featurefilter, 'function_handle'),
  opt.featurefilter = {{opt.featurefilter}};
elseif iscell(opt.featurefilter) & ~iscell(opt.featurefilter{1}),
  % filter of the form {'string', ...}
  opt.featurefilter = {opt.featurefilter};
end
if length(opt.featurethresh)==1,
  opt.featurethresh = repmat(opt.featurethresh, size(opt.featurefilter));
end

selected = logical(zeros([dim 1]));
fv = struct('x', Xtrain, 'y', Ytrain);
if opt.verbosity>0,
  fprintf('Option ''featurefilter'': Starting to filter...\n');
end
for i = 1:length(opt.featurefilter),
  [f, param] = getFuncParam(opt.featurefilter{i});
  p = feval(f, fv, param{:});
  selected = selected | (p<=opt.featurethresh(i));
  if opt.verbosity>0,
    fprintf('Feature filter #%i selected a subset of %2i features.\n', i, ...
            nnz(p<=opt.featurethresh(i)));
  end
end
if opt.verbosity>0,
  fprintf('Option ''featurefilter'' selected a total of %2i features.\n', ...
          nnz(selected));
end
Xtrain = Xtrain(selected,:);

if isfield(opt, 'kernelindex'),
  for i = 1:length(opt.kernelindex),
    j = opt.kernelindex{i};
    opt.kernelindex{i} = j(selected(j));
  end
end
if opt.verbosity>0,
  fprintf('Option ''featurefilter'': Feature subsets have been selected. It may\n');
  fprintf('now be necessary to adapt kernel parameters that are feature specific, but\n');
  fprintf('this can not be done automagically.\n');
end
