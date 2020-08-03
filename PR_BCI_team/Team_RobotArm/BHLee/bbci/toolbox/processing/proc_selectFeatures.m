function fv= proc_selectFeatures(fv, idx)
% proc_selectFeatures - Select subsets of features by index or name
%
% Synopsis:
%   fv = proc_selectFeatures(fv,idx)
%   
% Arguments:
%  fv: Feature vector structure, required field is 'x' ([dim N] data
%      matrix), optional field 'features'
%  idx: Index vector (select the feature with the given index) or cell
%      string (select the features with given name)
%   
% Returns:
%  fv: Feature vector structure with field x now of dimension
%      [length(idx) N]
%   
% Description:
%   Selecting a feature subset amounts to selecting a subset of rows from
%   field 'x' of a feature vector structure fv.
%   Named feature selection requires the fv to have a field 'features'
%   containing the string names of each feature.
%   Note that for EEG data, features are channels. EEG data usually has
%   fields 'clab' (channel label). proc_selectChannels is the EEG data
%   equivalent of proc_selectFeatures.
%   
%   
% Examples:
%   proc_selectFeatures(fv, [1 2 4]);
%     returns an updated feature vector structure with new field 'x'
%     equal to fv.x([1 2 4], :).
%   If fv has a field fv.features={'a', 'b', 'c'}:
%     proc_selectFeatures(fv, {'a', 'c'});
%   will return the first and third feature.
%
%   
% See also: proc_selectChannels
% 

% Author(s), Copyright: Anton Schwaighofer, Oct 2005
% $Id: proc_selectFeatures.m,v 1.1 2005/11/20 18:18:40 neuro_toolbox Exp $

error(nargchk(2, 2, nargin));

if ischar(idx),
  idx = {idx};
end
if iscell(idx),
  if ~isfield(fv, 'features'),
    error('With named feature indexing, fv must have a field ''features''');
  end
  [dummy, idx] = ismember(idx, fv.features);
  if any(idx==0),
    warning('Some features listed in idx are not present in fv.features.');
    idx = idx(idx~=0);
  end
end
if ndims(fv.x)>2,
  error('Ooops, this is currently only implemented for 2-dimensional fv.x');
end
fv.x = fv.x(idx,:);
if isfield(fv, 'features'),
  fv.features = fv.features(idx);
end
