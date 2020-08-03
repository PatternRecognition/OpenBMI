function [epo1,W] = proc_transformFeatures(epo1,W,dim)
%proc_transformFeatures tranforms by pseudo-inverse techniques one
%feature in another.
%
% description: 
%  after transforming into matrices by the information dim a matrix
%  W is found with W*epo1~epo2 (W=epo2*pinv(epo1)). W and
%  additionally epo=W*epo1 is given back
%
% usage:
%     [epo,W] = proc_transformFeatures(epo1,epo2,dim);
%     [epo,W] = proc_transformFeatures(epo1,W,dim);
%     In the second case the matrix W is used to calculate
%     epo=W*epo1.
%
% input:
%     epo1     - a usual epo structure with field x 
%     epo2     - a usual epo structure with field x or a matrix W
%                (the fields x in epo1 and epo2 must have the same 
%                dimension)
%     dim      - the dimensions which are used as rows in the
%                description (product of the dimensions of the
%                used dimensions equal to size(W)). e.g. if dim=1
%                epo1.x will be transformed to a matrix with
%                size(.,1)=size(epo.x,1), if dim =[1,2] to a matrix
%                with size(.,1) = size(epo.x,1)*size(epo.x,2)
%                default dim = [1,...,n-1] where n is the number of
%                dimensions epo1.x  has.
%
% output:
%     epo      - the tranformed epo1 structure
%     W        - the used transformation matrix W
%
% Guido Dornhege, 19/09/2003

nd = ndims(epo1.x);

if ~exist('dim','var') | isempty(dim)
  dim = 1:nd-1;
end

sx = size(epo1.x);
odim = setdiff(1:nd,dim);

epo1.x = permute(epo1.x,[dim,odim]);
epo1.x = reshape(epo1.x,[prod(sx(dim)),prod(sx(odim))]);

if isstruct(W)
  W = permute(W.x,[dim,odim]);
  W = reshape(W,[prod(sx(dim)),prod(sx(odim))]);
  W = W*pinv(epo1.x);
end

epo1.x = W*epo1.x;
epo1.x = reshape(epo1.x,[sx(dim),sx(odim)]);
epo1.x = ipermute(epo1.x,[dim,odim]);

