function [epo,w,la] = proc_multicsp_sim(epo,nComps,centered,method,choice);
% PROC_MULTICSP summarize all CSP methods so far, multi-class 
% works by simultaneous diagonalisation
%
% usage:
%   [epo,w,la] = proc_multicsp(epo,nComps,centered,method,choice);
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
%   choice       A string which describes the order of taken
%                eigenvectors concording the corresponding
%                eigenvalues. It consists of the chars 'h' and 'l'
%                and is repeated so often that it has at least
%                nComps chars. For example: if you write 'h' and
%                nComps is 3 it is used as 'hhh', 'hl' is used as
%                'hlh',... THe patterns are chosen regarding these
%                strings for each class corresponding the highest
%                resp. lowest eigenvalue. Default: 'h'. 
%                NOTE: For 2-classes these value is automatically
%                set to s
%                Further a string 's' is possible, which use for
%                each class the pattern to the highest value of max(ev,1-ev).
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

if ~exist('choice','var') | isempty(choice)
  choice = 's';
end

choice = repmat(choice,[1,nComps]);
choice = choice(1:nComps);

if ~isempty(findstr(choice,'s'))
  choice = 's';
end

if ~isempty(findstr(choice,'p'))
	choice = 'p';
end

if ~isempty(findstr(choice,'o'))
	choice = 'o';
end

if ~isempty(setdiff(choice,'hlspo'))
   error([mfilename ': 5th argument wrong']);
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

%if nClasses == 2
%  choice = 's';
%end


Sig = zeros(nChan,nChan,nClasses);
for i = 1:nClasses
  da = dat(:,:,find(labels(i,:)>0));
  da = da(:,:);
  if centered
    da = da-repmat(mean(da,2),[1 size(da,2)]);
  end
  Sig(:,:,i) = da*da'/size(da,2);
end

% do the simultaneuos diagonalisation
if (nClasses) == 2  % for two class there is a simple algorithm
  [P,D] = eig(sum(Sig,3));
  
  P = P*diag(sqrt(max(1./diag(D),0)));
  Sig = P'*Sig(:,:,1)*P;
  Sig = 0.5*(Sig+Sig');
  [R,D] = eig(Sig);
  V = P*R;
  D = cat(3,D,1-D);
else
  [V,moni] = simDiag(Sig,'fastdiag');
  if isfield(moni,'diags')  
    D = moni.diags;
  else
    D = zeros(size(Sig));
    for i = 1:nClasses
      D(:,:,i) = V*Sig(:,:,i)*V';
    end
  end
  DD = sum(D,3);
  V = V'*diag(sqrt(max(1./diag(DD),0)));
  for i = 1:nClasses
    D(:,:,i) = V'*Sig(:,:,i)*V;
  end
end

% only recognise relevant patterns
relind = find(diag(sum(D,3))>0 & diag(prod(D>=0,3))>0); 


% for chosing patterns the silent and smoothy opts has to
% been recognised
Ddh = zeros(length(relind),length(relind));
Ddl = zeros(length(relind),length(relind));

Dc = D(relind,relind,1:nClasses);
if strcmp(choice,'s') | strcmp(choice,'p')
  Dch = max(Dc,1./(1+(Dc./(1-Dc))*(nClasses-1)^2));
  hi = 1:nComps;
  nhi = length(hi);
elseif strcmp(choice,'o') 
  Dch = Dc;
  hi = 1:nComps;
  nhi = length(hi);
else
  Dch = Dc;
  Dcl = Dc;
  hi = findstr(choice,'h');
  lo = findstr(choice,'l');
  nhi = length(hi);
  nlo = length(lo);
  indil = zeros(nClasses,length(relind));
  lal = zeros(nClasses,length(relind));
end
% now choose the patterns
indih = zeros(nClasses,length(relind));
lah = zeros(nClasses,length(relind));
laha = zeros(nClasses,length(relind));



for i = 1:nClasses
  if ~strcmp(choice,'p') & ~strcmp(choice,'o')
    DD = diag(Dch(:,:,i));
    [DD,I] = sort(DD);
    indih(i,:) = I(end:-1:1)';
    lah(i,:) = diag(Dc(indih(i,:),indih(i,:),i))';
    laha(i,:) = diag(Dch(indih(i,:),indih(i,:),i))';
    %  for k = 1:optfunc
    %      la(hi,nClasses+k) = diag(D(ind(hi,i),ind(hi,i),k+nClasses));
    %    end
  else
    laha(i,:) = diag(Dch(:,:,i))';
    lah(i,:) = diag(Dc(:,:,i))';
    indih(i,:) = 1:length(relind);
  end
  if ~strcmp(choice,'s') & ~strcmp(choice,'p') & ~strcmp(choice,'o')
    DD = diag(Dcl(:,:,i));
    [DD,I] = sort(DD);  
    indil(i,:) = I(1:end)';
    lal(i,:) = diag(Dc(indil(i,:),indil(i,:),i))';
%    for k = 1:optfunc
%      la(lo,nClasses+k) = diag(D(ind(lo,i),ind(lo,i),k+nClasses));
%    end
  end
end


% choose the highest
if strcmp(choice,'s')
  IND = [];
  LA = [];
  i = 1;
  while i<=nhi
    l = indih(:,i);
    ll = intersect(l,IND(:));
    le = unique(l);
    if ~isempty(ll)
      for j = 1:nClasses
	aa = find(l(j)==ll);
	if ~isempty(aa)
	  indih(j,i:end-1) = indih(j,i+1:end);
	  indih(j,end) = 0;
	  lah(j,i:end-1) = lah(j,i+1:end);
	  lah(j,end) = 0;
	  laha(j,i:end-1) = laha(j,i+1:end);
	  laha(j,end) = 0;
	end
      end
    elseif length(le)~=nClasses
      for j = 1:length(le)
	aa = find(le(j)==l);
	if length(aa)>1
	  [dum,pl] = max(laha(aa,i));
	  pl = aa([1:pl-1,pl+1:end]);
	  indih(pl,i:end-1) = indih(pl,i+1:end);
	  indih(pl,end) = 0;
	  lah(pl,i:end-1) = lah(pl,i+1:end);
	  lah(pl,end) = 0;
	  laha(pl,i:end-1) = laha(pl,i+1:end);
	  laha(pl,end) = 0;
	end
      end
    else
      IND = [IND,indih(:,i)];
      LA = [LA,lah(:,i)];
      i = i+1;
    end
  end
elseif strcmp(choice,'p')
  [LAHAa,INDa] = max(laha,[],1);
  [dum,india] = sort(-LAHAa);
  INDa = INDa(india);
  for bla = 1:nClasses
    in = find(INDa==bla);
    in = india(in(1:min(length(in),nComps)));
    IND(bla,1:length(in)) = in;
    LA(bla,1:length(in)) = lah(bla,in);
  end  
elseif strcmp(choice,'o')
  [maxLAH,maxIND] = max(laha,[],1);
  [minLAH,minIND] = min(laha,[],1);
  maxdif = sum(abs(laha-repmat(maxLAH,[nClasses,1])),1);
  mindif = sum(abs(laha-repmat(minLAH,[nClasses,1])),1);
  minin = find(mindif>maxdif);
  maxdif(minin) = mindif(minin);
  maxIND(minin) = minIND(minin);
  [dum,india] = sort(-maxdif);
  INDa = maxIND(india);
  for bla = 1:nClasses
    in = find(INDa==bla);
    in = india(in(1:min(length(in),nComps)));
    IND(bla,1:length(in)) = in;
    LA(bla,1:length(in)) = lah(bla,in);
  end    
else
  IND = zeros(nClasses,nComps);
  IND(:,hi) = indih(:,1:nhi);
  IND(:,lo) = indil(:,1:nlo);
  LA = zeros(nClasses,nComps);
  LA(:,hi) = lah(:,1:nhi);
  LA(:,lo) = lal(:,1:nlo);
end

in = find(IND~=0);
INDI = IND(in);
LA = LA(in);

ind = IND';
la = LA';

w = V(:,INDI);

% and project
epo = proc_linearDerivation(epo,w);

% cosmetic correction
epo.clab = cell(1,length(INDI));
for i = 1:length(INDI)
  [I,J] = find(INDI(i)==ind);
  epo.clab{i} = sprintf('Class %s, Pattern %i',epo.className{J(1)}, I(1));
  for j = 2:length(I)
    epo.clab{i} = sprintf('%s;Class %s, Pattern %i',epo.clab{i},epo.className{J(j)}, I(j));
  end
end

  
