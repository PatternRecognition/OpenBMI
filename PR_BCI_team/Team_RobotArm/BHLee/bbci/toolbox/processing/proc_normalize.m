function [fv, opt]= proc_normalize(fv, varargin)
%[fv, opt]= proc_normalize(fv, opt)
%
% IN  fv    - struct of feature vectors
%     opt  struct and/or property/value list of properties
%      .policy - 'std' (default), 'max', 'norm', 'nanstd'. max : normalizing abs-max 
%                to 1, norm: normalizing euclidean norm to 1,
%                std: normalizing std to 1.
%      .dim    - dimension along which fv should be normalized,
%                1 is normalizing each sample (feature vector),
%                2 is normalizing each feature dimension of fv (default).
%      .scale  - vector by which fv is scaled. Typically this is calculated
%                by this function. This field can be used to apply the
%                scaling calculated from one data set (e.g. training data)
%                to another data set (e.g. test data)
%
%      fv   - struct of scaled feature vectors
%      opt  - as input but with new field .scale
%
% You can use the short form fv= proc_normalize(fv, <policy>).
%
% NOTE: proc_nomrlaize DOES NOT subtract the mean. Use proc_subtractMean
% for that.
%
% See also: nanstd
%

% bb 09/03, ida.first.fhg.de
% Anton Schwaighofer, Feb 2005

if length(varargin)==1 & ischar(varargin{1}),
  opt= struct('policy', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'policy', 'std', ...
                  'dim', 2);

sz= size(fv.x);

if isfield(opt, 'scale'),
  ssz= sz;
  ssz(opt.dim)= 1;
  if ~isequal(size(opt.scale), ssz),
    error('size of opt.scale does not match fv');
  end
else
  switch(opt.policy),
   case 'std',
    opt.scale= std(fv.x, 0, opt.dim);
   case 'norm',
    opt.scale= sqrt(sum(fv.x.^2, opt.dim));
   case 'max',
    opt.scale= max(abs(fv.x), [], opt.dim);
   case 'nanstd',
    opt.scale= nanstd(fv.x, 0, opt.dim);
  end
  iz= find(opt.scale==0);
  opt.scale(iz)= 1;
  opt.scale= 1./opt.scale;
end

%% scaling that works with more than 2 dimensions
rep_sz= ones(1, max(length(sz), opt.dim));
rep_sz(opt.dim)= sz(opt.dim);

fv.x= fv.x .* repmat(opt.scale, rep_sz);


%% scaling that works only with 2 dimensions
%srt= [opt.dim 3-opt.dim)];
%xx= permute(fv.x, srt);
%xx= xx * diag(opt.scale);
%fv.x= ipermute(xx, srt);
