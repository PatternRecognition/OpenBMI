function opt = gaussProc_subsetParams(opt, idx)
% gaussProc_subsetParams - Parameter subsets for data subsets
%
% Synopsis:
%   opt = gaussProc_subsetParams(opt,idx)
%   
% Arguments:
%  opt: GP options structure
%  idx: Data indices or logical index
%   
% Returns:
%  opt: Options for data subset
%   
% Description:
%   Compute subsets for all parameters that are specific to data
%   points. Currently, this is only parameter 'noisegroups'
%   
% See also: 
% 

% Author(s), Copyright: Anton Schwaighofer, Nov 2005
% $Id: gaussProc_subsetParams.m,v 1.2 2007/04/23 11:22:26 neuro_toolbox Exp $


% There is only one parameter that is specific to examples:
% noisegroups. Need to build subsets for that
ng = opt.noisegroups;
for j = 1:length(ng),
  if islogical(ng{j}),
    ng{j} = ng{j}(idx);
  else 
    if islogical(idx),
      idx = find(idx);
    end
    ng{j} = intersect(ng{j}, idx);
  end
end
opt.noisegroups = ng;
