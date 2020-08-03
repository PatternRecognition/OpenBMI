function [epo,w,la] = proc_multicsp(epo,nComps,centered,method,way);
% PROC_CSP_COMMON summarize all CSP methods so far, additionally
% Kernel methods can be used, and multi-class works by simultaneous
% diagonalisation
%
% usage:
%   [epo,w,la] = proc_csp_common(epo,nComps,centered,method,way);
%
% input:
%   epo          a usual epo structure with field x (nTimes,nChan,nTrials) 
%                and field y for the labels
%                if field optimise is given, it is added to
%                variable optimise in this call
%   nComps       the number of Patterns for each class [1]
%   centered     a flag for centered covariances (1), or
%                non-centered covariance (0), [1]
%   method       a string which describes the handling of each time
%                point by calculating the covariance. It can be 
%                'all'  all time points are used as sample
%                'mean' the mean over time in each trial is used as
%                sample   
%                default: 'all' for the centered case, 'mean' for
%                the non-centered case
%   way          'one-vs-all','pairwise','complete'
%
% output:
%   epo          the projected epo structure
%   w            the projection patterns
%   la           the corresponding eigenvalues as a matrix
%                nComps*nClasses
%
% Guido Dornhege, 24/03/03

% check input, set defaults

if ~exist('nComps','var') | isempty(nComps)
  nComps = 1;
end

if ~exist('centered','var') | isempty(centered)
  centered = 1;
end

if ~exist('method','var') | isempty(method)
  if centered 
    method = 'all';
  else
    method = 'mean';
  end
end

if ~exist('way','var') | isempty(way)
  way = 'one-vs-all';
end

nClasses = size(epo.y,1);
if nClasses==1
  epo.y = [epo.y<0;epo.y>0];
  nClasses == 2;
end

if ischar(way)
  
  switch way
   case 'one-vs-all'
    way = 2*eye(nClasses)-ones(nClasses);
   case 'pairwise'
    way = [];
    for i = 1:nClasses
      for j = i+1:nClasses;
	vec = zeros(1,nClasses);
	vec(i) = 1;
	vec(j) = -1;
	way = [way;vec];
      end
    end
   case 'complete'
    a = dec2bin(1:2^(nClasses-1)-1,nClasses);
    way = [];
    for i = 1:size(a,1)
      d = transpose(str2num(a(i,:)'));
      c = sum(d);
      if c>=2
	b = dec2bin(1:2^(c-1)-1,c);
	for j = 1:size(b,1)
	  bb =  transpose(str2num(b(j,:)'));
	  e = d;
	  e(find(d>0))=2*bb-1;
	  way = [way;e];
	end
      end
    end
    
  end
end

if size(epo.y,1)==1
  cl = unique(epo.y);
  labels = zeros(length(cl),size(epo.y,2));
  for j = 1:length(cl)
    labels(j,find(epo.y==cl(j)))=1;
  end
else
  labels = epo.y;
end

% calculate the samples

[nTimes,nChan,nTrials] = size(epo.x);
dat = permute(epo.x,[2 1 3]);

switch method
 case 'all'
  % nothing to do
 case 'mean'
  dat = mean(dat,2);
 otherwise
  error([mfilename ': 4th argument wrong']);
end

% Calculate the covariance matrices
nFeat = size(dat,2)*size(dat,3);

nClasses = size(labels,1);

if nComps*nClasses>=nChan
  error([mfilename ': Projection make no sense, to many' ...
		    ' patterns']);
end

Sig = zeros(nChan,nChan,nClasses);
for i = 1:nClasses
  da = dat(:,:,find(labels(i,:)>0));
  da = da(:,:);
  if centered
    da = da-repmat(mean(da,2),[1 size(da,2)]);
  end
  Sig(:,:,i) = da*da'/size(da,2);
end

w = zeros(nChan,size(way,1),nComps*2);
la = zeros(size(way,1),nComps*2);
% do the simultaneuos diagonalisation
for i = 1:size(way,1)
  ind1 = find(way(i,:)==1);
  ind2 = find(way(i,:)==-1);
  Sig1 = mean(Sig(:,:,ind1),3);
  Sig2 = mean(Sig(:,:,ind2),3);
  
  [P,D] = eig(Sig1+Sig2);
  
  P = P*diag(sqrt(max(1./diag(D),0)));
  Sig1 = P'*Sig1*P;
  Sig1 = 0.5*(Sig1+Sig1');
  [R,D] = eig(Sig1);
  [lam,ind] = sort(diag(D));
  lam = lam([1:nComps,end-nComps+1:end]);
  ind = ind([1:nComps,end-nComps+1:end]);
  R = R(:,ind);
  V = P*R;
  la(i,:) = lam';
  w(:,i,:) = reshape(V,[nChan,1,2*nComps]);
end


la = la';
la = la(:);

w = w(:,:);

epo = proc_linearDerivation(epo,w);


