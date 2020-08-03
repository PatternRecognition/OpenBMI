function [fv, opt]= proc_subtractMean(fv, varargin)
%[fv, opt]= proc_subtractMean(fv, opt)
%
%  Subtract mean or median from data, given in fv.
%
% IN  fv    - struct of feature vectors
%     opt  struct and/or property/value list of properties
%      .policy - one of 'mean' (default), 'median', 'min', 'nanmean', 'nanmedian'
%      .dim    - dimension along which this function operates. Default
%                value: 2
%      .bias   - vector which is subtracted from fv. If this option is
%                given, the 'policy' option is ignored. 
%                Typically the bias vector is computed in a first call to
%                proc_subtractMean. Thus, this field can be used to 
%                apply the shift calculated from one data set (e.g. 
%                training data) to another data set (e.g. test data)
%
%      fv   - struct of shifted feature vectors
%      opt  - a copy of the input options, with a new field .bias that
%             contains the mean/median/man vector that has been
%             subtracted from the data.
%
% See also: nanmean

% bb 09/03, ida.first.fhg.de
% Anton Schwaighofer, Feb 2005

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'policy', 'mean', ...
                  'dim', 2);

sz= size(fv.x);

if isfield(opt, 'bias'),
  bsz= sz;
  bsz(opt.dim)= 1;
  if ~isequal(size(opt.bias), bsz),
    error('size of opt.bias does not match fv');
  end
else
  switch(opt.policy),
   case 'mean',
    opt.bias= mean(fv.x, opt.dim);
   case 'median',
    opt.bias= median(fv.x, opt.dim);
   case 'min',
    opt.bias= min(fv.x, [], opt.dim);
   case 'nanmean',
    opt.bias= nanmean(fv.x, opt.dim);
   case 'nanmedian',
    opt.bias= nanmedian(fv.x, opt.dim);
  end
end

rep_sz= ones(1, max(length(sz), opt.dim));
rep_sz(opt.dim)= sz(opt.dim);

fv.x= fv.x - repmat(opt.bias, rep_sz);
