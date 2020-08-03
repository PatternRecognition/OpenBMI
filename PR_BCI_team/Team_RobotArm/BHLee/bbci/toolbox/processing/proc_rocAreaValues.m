function fv_roc= proc_rocAreaValues(fv, varargin)
%PROC_ROCAREAVALUES - Measure of Discriminability based on ROC Curves
%
%For each feature dimension, the area under the ROC curve is calculated,
%and the rocAreaValue is determined as 2*(0.5-area), i.e., the value is
%-1 if all samples of class 1 have smaller values than any sample of 
%class 2, and the value is 1 if all samples of class 1 have larger 
%values than any sample of class 2.
%For data with more than two classes, pairwise roc values are calculated.
%
%Synopsis:
% FV_ROC= proc_rocAreaValues(FV)
%
%Arguments:
% FV: Feature vector structure
%
%Returns:
% FV_OUT: Feature vector structure of roc values.
%
%See also:
% proc_r_values, proc_r_squared_signed, proc_t_values

% Author(s): Benjamin Blankertz, Feb 2006

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'multiclass_policy', 'pairwise');

if size(fv.y,1)>2, 
  fv_roc= proc_wr_multiclass_diff(fv, {'rocAreaValues',opt}, ...
                                  opt.multiclass_policy);
  return;
end
  
sz= size(fv.x);
fv.x= reshape(fv.x, [prod(sz(1:end-1)) sz(end)]);

roc= zeros(size(fv.x,1), 1);
for k= 1:size(fv.x,1),
  [dmy, auc]= val_rocCurve(fv.y, fv.x(k,:));
  roc(k)= 2*(0.5-auc);
end

fv_roc= copy_struct(fv, 'not', 'x','y','className');
fv_roc.x= reshape(roc, [sz(1:end-1) 1]);
if isfield(fv, 'className'),
  fv_roc.className= {sprintf('roc( %s , %s )', fv.className{1:2})};
end
fv_roc.y= 1;
fv_roc.yUnit= 'roc';
