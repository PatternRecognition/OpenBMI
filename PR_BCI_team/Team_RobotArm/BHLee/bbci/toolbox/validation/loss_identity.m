function loss = loss_identity(label, out)
% LOSS_IDENTITY - Identity loss function (for likelihoods or alike)
%
%   This is a dummy function to, for example, allow the evaluation of
%   likelihoods in cross-validation procedures. 
%   LOSS = LOSS_IDENTITY(LABEL, OUT) simply returns the input argument
%   OUT. To evaluate likelihoods, the apply function of a classifier
%   would return a vector OUT of likelihoods, which are then passed on to
%   LOSS_IDENTITY by the cross-validation procedure.
%   
%   See also LOSS_0_1,XVALIDATION
%

% anton@first.fraunhofer.de

error(nargchk(2, 2, nargin));

loss = out;
