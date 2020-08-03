function fv= proc_r_square_signed(fv, varargin)
%fv_rsqu= proc_r_square_signed(fv, <opt>)
%
% Computes the r^2 value for each feature, multiplied by the sign of
% of r value. The r^2 value is a measure
% of how much variance of the joint distribution can be explained by
% class membership.
% Any suggestion for a good reference?
%
% IN   fv  - data structure of feature vectors
%      opt - struct or property/value list of optional properties
%       .tolerate_nans: observations with NaN value are skipped
%            (nanmean/nanstd are used instead of mean/std)
%
% OUT  fv_rsqu - data structute of signed r^2 values (one sample only)
%
% SEE  proc_t_scaled, proc_r_values, proc_r_square

% bb 03/2003, ida.first.fhg.de

if length(varargin)>0 & isstruct(varargin{1}),
  fv2= varargin{1};
  varargin= varargin(2:end);
  if size(fv.y,1)*size(fv2.y,1)>1,
    error('when using 2 data sets both may only contain 1 sigle class');
  end
  if strcmp(fv.className{1}, fv2.className{1}),
    fv2.className{1}= strcat(fv2.className{1}, '2');
  end
  fv= proc_appendEpochs(fv, fv2);
  clear fv2;
end

fv= proc_r_values(fv, varargin{:});
fv.x= fv.x .* abs(fv.x);
for cc= 1:length(fv.className),
  fv.className{cc}= ['sgn r^2' fv.className{cc}(2:end)];
end
fv.yUnit= 'sgn r^2';
