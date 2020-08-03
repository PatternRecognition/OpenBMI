function fv= proc_selectSamples(fv, idx)
%PROC_SELECTSAMPLES - Select a subset of samples of a data set
%
%Description:
% Select a subset of samples (feature vectors) from the feature vector 
% struct fv.
% If your samples are EEG epochs in a struct that has the field
% 'indexedByEpochs' please use the function proc_selectEpochs.
% 
%Usage:
% FV= xval_selectSamples(FV, IDX)
%
%Input:
% FV:  struct of feature vectors
% IDX: indices of the feature to select
%
%Output:
% FV: reduced set of feature vectors
%
%See also xvalidation, proc_selectEpochs

% bb ida.first.fhg.de 07/2004

nd= ndims(fv.x);
ii= repmat({':'}, [1 nd]);
ii{nd}= idx;

fv.x= fv.x(ii{:});
fv.y= fv.y(:,idx);

if isfield(fv, 'bidx'),
  fv.bidx= fv.bidx(idx);
end
if isfield(fv, 'jit'),
  fv.jit= fv.jit(idx);
end
