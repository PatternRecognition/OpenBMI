function fv_out= proc_wr_multiclass_diff(fv, diff_fnc, varargin)
%PROC_WR_MULTICLASS_DIFF - Wrapper function for multi-class differences
%
%Synopsis:
% FV_OUT= proc_wr_multiclass_diff(FV, FCN, <POLICY>)
% FV_OUT= proc_wr_multiclass_diff(FV, FCN, <OPT>)
%
%Arguments:
% FV: Feature vector structure
% FCN: Function which is called for pairwise differences
%      (this is typically the callind function).
%      The function can either be a string (then the function is called
%      without additional arguments), or a cell {FCN_NAME, FCN_PARAMS}
%      
% POLICY: Policy of doing the pairwise difference:
%      'pairwise' (default), 'each-against-last', 'each-against-rest',
%       OR a [nPairs x 2] sized matrix that explicitly specifies the pairs.
% OPT: Struct or property-value-list of optional properties. So far only
%   .policy: See above, argument POLICY.
%
%Returns:
% FV_OUT: Feature vector structure of roc values.
%
%See also:
% proc_rocAreaValues, proc_r_values, proc_r_squared_signed, proc_t_values

% Author(s): Benjamin Blankertz, Oct 2010


if length(varargin)==1 && ~isstruct(varargin{1}),
  opt= struct('policy', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'policy', 'pairwise');

[fcn, params]= getFuncParam(diff_fnc);

if ischar(opt.policy),
  switch(opt.policy),
   case 'pairwise'
    opt.policy= nchoosek(1:size(fv.y,1), 2);
   case {'all-against-last', 'each-against-last'},
    ni= size(fv.y,1)-1;
    opt.policy= [[1:ni]' (ni+1)*ones(ni,1)];
  end
end

fv_out= [];
if isnumeric(opt.policy),
  for ic= 1:size(opt.policy,1),
    ep= proc_selectClasses_keepVoids(fv, opt.policy(ic,:));
    fv_tmp= feval(['proc_' fcn], ep, params{:});
    fv_out = proc_appendEpochs(fv_out, fv_tmp);
  end
  return;
else
  switch(opt.policy),
   case 'each-against-rest',
    ni= size(fv.y,1);
    combs= zeros(ni,ni-1);
    for ic= 1:ni,
      combs(ic,:)= [1:ic-1, ic+1:ni];
    end
    for ic= 1:ni,
      ep= proc_combineClasses(fv, combs(ic,:));
      fv_tmp= feval(['proc_' fcn], ep, params{:});
      fv_out = proc_appendEpochs(fv_out, fv_tmp);
    end
    return; 
   otherwise,
    error('unknown multiclass policy');
  end
end
