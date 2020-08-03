function fv= xval_selectSamples(fv, idx)
%XVAL_SELECTSAMPLES - helper function for XVALIDATION
%
%Description:
% Select a subset of samples (feature vectors) from the feature vector 
% struct fv. Since this function is specifically designed for the use
% in XVALIDATION, it should not be used otherwise.
% Otherwise use the function proc_selectEpochs.
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


fv.y= fv.y(:,idx);

if ~isfield(fv, 'continuous'),
  nd= ndims(fv.x);
  ii= repmat({':'}, [1 nd]);
  ii{nd}= idx;
  fv.x= fv.x(ii{:});
end

if isfield(fv, 'pos'),
  fv.pos= fv.pos(idx);
end

if isfield(fv, 'bidx'),
  fv.bidx= fv.bidx(idx);
end
if isfield(fv, 'jit'),
  fv.jit= fv.jit(idx);
end
