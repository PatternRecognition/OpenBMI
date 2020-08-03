function [fv,W]=proc_whiten(fv, varargin)
% PROC_WHITEN - Perform whitening
%
%Synopsis:
% [FV,W]= proc_whiten(FV, 'Property',Value,...)
%
%Description:
% Perform whitening, i.e., linearly transform the data so that
% the sensor covariance matrix is identity. This functions works
% either for time series, and for covariance matrices (i.e. after
% proc_covariance has been called).
%
%Arguments:
%  FV : struct of continuous or epoched data (or covariance matrices).
%
%Returns:
%  FV : struct of whitened data (or covariance matrices).
%  W  : whitening matrix
%
%Properties:
% 'appendix': appendix to channel labels. 'wht' is the default.
% 'copyclab': Copy the channel labels (field .clab) from input to output
%    structure. Otherwise the original channel labels are stored in the
%    field .origClab. Default Value: 0.
%
%See also: proc_covariance

%Author(s): Ryota Tomika
%
% Oct 2006: the default channel name is e.g., 'C1 wht', 'C3 wht', 'Cz wht'...
% Aug 2006: modified to work for continuous data + copyclab prop (Benjamin)
% Jul 2006: modified so that W is symmetric

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'copyclab', 0,...
                  'appendix', 'wht');

[T,d,n]= size(fv.x);

%% index of valid samples
if isfield(fv, 'y'),
  iValid= find(any(fv.y,1));
else
  %% this is for continuous data
  iValid= 1;
end

%% When the first dimension is singleton, the function
%% assumes that proc_covariance has been already done.

if ~isfield(fv,'clab')
  fv.clab = [];
end


if T==1
  d= sqrt(d);
  V= reshape(fv.x, [d,d,n]);
else
  V= zeros(d,d,n);
  for i= 1:n,
    V(:,:,i)= cov(fv.x(:,:,i));
  end
end

Sigma = mean(V(:,:,iValid),3);
[EV, ED]=eig(Sigma);
W = EV*diag(1./(sqrt(diag(ED))))*EV';

if T==1,
  for i= 1:n,
    Vi= W'*V(:,:,i)*W;
    V(:,:,i)= (Vi+Vi')/2;
  end
  fv.x= reshape(V, [1,d*d,n]);
else
  fv.origClab= fv.clab;
  fv= proc_linearDerivation(fv, W, 'clab', fv.clab);
end

if ~opt.copyclab
  for ic=1:length(fv.clab)
    fv.clab{ic} = [fv.clab{ic} ' ' opt.appendix];
  end
end
