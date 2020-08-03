function [S,n,className] = proc_get_covariances(epo,varargin)
% Sigma = proc_get_covariances(epo,<opt>)
%
% Calculate classwise covariances with respective numbers of trials.
%
% IN:  epo    - struct of epoched EEG data.
% OUT: S      - covariances channel x channel x classes
%      n      - numbers of trials per class.
%      className - name of the corresponding classes.

% kraulem 07/06

if nargin>1
  opt = propertylist2struct(varargin{:});
else
  opt = struct;
end

opt = set_defaults(opt,'classwise',1);

nClasses = size(epo.y,1);
nChans = size(epo.x,2);

switch opt.classwise
  case 1
   % find the covariance matrix for each class.
   S = zeros(nChans,nChans,nClasses);
   className = epo.className;
   for cl = 1:nClasses
     C = zeros(nChans,nChans);
     cl_ind = find(epo.y(cl,:));
     for mm = cl_ind
       C = C + cov(epo.x(:,:,mm));
     end
     n(cl) = length(cl_ind);
     S(:,:,cl) = C/n(cl);
   end
 case 0
  % find the covariance matrix of the entire EEG.
  S = zeros(nChans,nChans);
  for mm = 1:size(epo.x,3)
    S = S + cov(epo.x(:,:,mm));
  end
  S = S/size(epo.x,3);
  n = size(epo.x,3);
  className = {sprintf('%s ',epo.className{:})};
 case 2
  % calculate covariances classwise, then average.
  S = proc_get_covariances(epo,'classwise',1);
  S = squeeze(sum(S,3));
  n = size(epo.x,3);
  className = {sprintf('%s ',epo.className{:})};
end
return