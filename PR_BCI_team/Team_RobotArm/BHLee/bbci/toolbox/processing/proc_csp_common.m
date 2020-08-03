function [epo,w,la] = proc_csp_common(epo,nComps,centered,method,optimise,choice,kernel,varargin)
% PROC_CSP_COMMON summarize all CSP methods so far, additionally
% Kernel methods can be used, and multi-class works by simultaneous
% diagonalisation
%
% usage:
%   [epo,w,la] = proc_csp_common(epo,nComps,centered,method,optimise,choice,kernel,params);
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
%   optimise     a struct with the following entries:
%                dat: can be a matrix where something (silent)
%                     should happen or 
%                     an interval, then mat is chosen out of epo.x
%                     regarding epo.t or
%                     an cell array of things like the above
%                method: a method out of 'smooth', 'silent' (last
%                        default), where smooth try to smooth the dats, silent
%                        try to minimize power in dat
%                influence: a cell {factor,approach}, where factor
%                           is a scalar or double which weights the
%                           influence of the dat for highest and
%                           lowest eigenvalues, approach is 
%                           'add', for adding the further
%                           constraint. Different method not
%                           implemented so far
%                           it can only be the factor, default: 1
%                centered: a cell array of flags
%                label: if  a label field is given to a tensor data
%                       matrix, and centered is set to 1, then
%                       centered covariances are calculated to the
%                       means of each class. 
%                if this variable is not given, and further no field
%                silent in epo is known, no further optimisiation
%                takes place    
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
%   kernel       a kernel function. By calling the string, it has
%                to calculate a matrix out of matrices x,y (collection of
%                column vectors) where the rows run over x, the
%                columns over y. It has to have the two matrices as
%                first input arguments. Further arguments are given
%                to the function by params in this function. The
%                following kernels are defined:
%                'linear': no kernel (default)
%                'poly'  : polynomial kernel and p [2] and c [0] is 
%                          an additional argument
%                'gauss' : gaussian kernel and sigma [1] is an
%                          additional argument
%   params       parameters for kernel, default are defined there.
% NOTE: KERNEL METHODS DON'T WORK SO FAR!
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

if ~exist('optimise','var') | isempty(optimise)
  optimise.dat = {};
  optimise.method = {};
  optimise.influence = {};
  optimise.centered = {};
end

if ~isstruct(optimise) 
  optimise.dat = optimise;
end

if ~iscell(optimise.dat);
  optimise.dat = {optimise.dat};
end

if ~isfield(optimise,'method')
  optimise.method = cell(1,length(optimise.dat));
end
if ~isfield(optimise,'influence')
  optimise.influence = cell(1,length(optimise.dat));
end
if ~isfield(optimise,'centered')
  optimise.centered = cell(1,length(optimise.dat));
end

if ~iscell(optimise.method)
  optimise.method = {optimise.method};
end
if ~iscell(optimise.influence)
  optimise.influence = {optimise.influence};
end
if ~iscell(optimise.centered)
  optimise.centered = {optimise.centered};
end

optfunc = length(optimise.dat);

if optfunc>0
  par = cell(1,optfunc);
  par(1:length(optimise.method)) = optimise.method;
  optimise.method = par;
  par = cell(1,optfunc);
  par(1:length(optimise.influence)) = optimise.influence;
  optimise.influence = par;
  par = cell(1,optfunc);
  par(1:length(optimise.centered)) = optimise.centered;
  optimise.centered = par;
end


if ~isfield(epo,'optimise') | isempty(epo.optimise)
  optim.dat = {};
  optim.method = {};
  optim.influence = {};
  optim.centered = {};
else 
  optim = epo.optimise;
end

if ~isstruct(optim) 
  optim.dat = optim;
end

if ~iscell(optim.dat);
  optim.dat = {optim.dat};
end

if ~isfield(optim,'method')
  optim.method = cell(1,length(optim.dat));
end
if ~isfield(optim,'influence')
  optim.influence = cell(1,length(optim.dat));
end
if ~isfield(optim,'centered')
  optim.centered = cell(1,length(optim.dat));
end

if ~iscell(optim.method)
  optim.method = {optim.method};
end
if ~iscell(optim.influence)
  optim.influence = {optim.influence};
end
if ~iscell(optim.centered)
  optim.centered = {optim.centered};
end

optfunc = length(optim.dat);

if optfunc>0
  par = cell(1,optfunc);
  par(1:length(optim.method)) = optim.method;
  optim.method = par;
  par = cell(1,optfunc);
  par(1:length(optim.influence)) = optim.influence;
  optim.influence = par;
  par = cell(1,optfunc);
  par(1:length(optim.centered)) = optim.centered;
  optim.centered = par;
end

optimise.dat = {optimise.dat{:},optim.dat{:}};
optimise.method = {optimise.method{:},optim.method{:}};
optimise.influence = {optimise.influence{:},optim.influence{:}};
optimise.centered = {optimise.centered{:},optim.centered{:}};


optfunc = length(optimise.dat);

for i = 1:optfunc
  if isempty(optimise.method{i})
    optimise.method{i} = 'silent';
  end
  if isempty(optimise.influence{i})
    optimise.influence{i} = 1;
  end
  if ~iscell(optimise.influence{i})
    optimise.influence{i} = {optimise.influence{i}};
  end
  if length(optimise.influence{i})==1
    optimise.influence{i} = {optimise.influence{i}{:},'add'};
  end
  if length(optimise.influence{i}{1})==1
    optimise.influence{i}{1} = optimise.influence{i}{1}*ones(1,2);
  end
  if isempty(optimise.centered{i})
    optimise.centered{i} = centered;
  end
end
  
if ~exist('choice','var') | isempty(choice)
  choice = 'h';
end

choice = repmat(choice,[1,nComps]);
choice = choice(1:nComps);

if ~isempty(findstr(choice,'s'))
  choice = 's';
end

  
if ~isempty(setdiff(choice,'hls'))
   error([mfilename ': 5th argument wrong']);
end

if ~exist('reduction','var') | isempty(reduction)
  reduction = 0;
end


if ~exist('kernel','var') | isempty(kernel)
  kernel = 'linear';
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

if nClasses == 2
  choice = 's';
end

if strcmp('linear',kernel)
  Sigma = zeros(nChan,nChan,nClasses);
  for i = 1:nClasses
    da = dat(:,:,find(labels(i,:)>0));
    da = da(:,:);
    if centered
      da = da-repmat(mean(da,2),[1 size(da,2)]);
    end
    Sigma(:,:,i) = da*da'/size(da,2);
  end
else
  % Kernel case
  Sigma = zeros(nFeat,nFeat,nClasses);
  % calculate Kernel matrix
  K = feval(kernel,dat(:,:),dat(:,:),varargin{:});
  for i = 1:nClasses
    ind = find(labels(i,:));
    %   KK = zeros(nFeat,nFeat);
    %   for j = ind
    %     KK = KK+K(:,j)*K(j,:)/length(ind);
    %   end
    KK = K*K'/length(ind);
    if centered
      for j=ind
	for k = ind
	  KK = KK - K(:,j)*K(k,:)/length(ind)/length(ind);
	end
      end
    end
    Sigma(:,:,i) = KK;
  end
end


Ofunc = zeros(size(Sigma,1),size(Sigma,2),optfunc);
% now calculate the matrices for further optimisation
for j = 1:optfunc
  mat = optimise.dat{j};
  if size(mat,1)==1 & size(mat,2)<=2 & size(mat,3)==1
    mat = proc_selectIval(epo,mat);
    mat = mat.x;
  end
  met = optimise.method{j};
  mat = permute(mat,[2 1 3]);
  switch met
   case 'silent'
    mat = mat(:,:);
   case 'smooth'
    mat = mat(:,1:end-2,:)-2*mat(:,2:end-1,:)+mat(:,3:end,:);
    mat = mat(:,:);
  end
  if optimise.centered{j}>0
    if isfield(optimise,'labels')
      for i = 1:size(labels,1)
	ind = find(optimise.labels(i,:));
	me = mean(mat(:,:,ind),2);
	mat(:,:,ind) = mat(:,:,ind)-repmat(me,[1,length(ind)]);
      end
    end
  end
  
  Ofunc(:,:,j) = mat*mat'/size(mat,2);
end

Sig = cat(3,Sigma,Ofunc);
% do the simultaneuos diagonalisation
if (nClasses+optfunc) == 2  % for two class there is a simple algorithm
  [P,D] = eig(sum(Sig,3));
  
  P = P*diag(sqrt(max(1./diag(D),0)));
  Sig = P'*Sig(:,:,1)*P;
  Sig = 0.5*(Sig+Sig');
  [R,D] = eig(Sig);
  V = P*R;
  D = cat(3,D,1-D);
else
  [V,moni] = simDiag(Sig,'phamdiag');
  if isfield(moni,'diags')  
    D = moni.diags;
  else
    D = zeros(size(Sig));
    for i = 1:nClasses+optfunc
      D(:,:,i) = V*Sig(:,:,i)*V';
    end
  end
  DD = sum(D,3);
  V = V'*diag(sqrt(max(1./diag(DD),0)));
  for i = 1:nClasses+optfunc
    D(:,:,i) = V'*Sig(:,:,i)*V;
  end
end

% only recognise relevant patterns
relind = find(diag(sum(D,3))>0 & diag(prod(D>=0,3))>0); 


% for chosing patterns the silent and smoothy opts has to
% been recognised
Ddh = zeros(length(relind),length(relind));
Ddl = zeros(length(relind),length(relind));
for i = 1:optfunc
  switch optimise.influence{i}{2}
   case 'add'
    Ddh = Ddh+ optimise.influence{i}{1}(1)*D(relind,relind,i+nClasses);
    Ddl = Ddl+ optimise.influence{i}{1}(2)*D(relind,relind,i+nClasses);
   otherwise
    error([mfilename ': optimise.influence should have a second argument']);
  end
end

Dc = D(relind,relind,1:nClasses);
if strcmp(choice,'s')
  Dch = max(Dc,(1-Dc)./(1+nClasses*(nClasses-1)*Dc));
  hi = 1:nComps;
  nhi = length(hi);
  if optfunc>0
    switch optimise.influence{i}{2}
     case 'add'
      Dch = Dch-repmat(Ddh,[1 1 nClasses]);
    end
  end
else
  Dch = Dc;
  Dcl = Dc;
  hi = findstr(choice,'h');
  lo = findstr(choice,'l');
  nhi = length(hi);
  nlo = length(lo);
  indil = zeros(nClasses,length(relind));
  lal = zeros(nClasses,length(relind));
  if optfunc>0
    switch optimise.influence{i}{2}
     case 'add'
      Dch = Dch-repmat(Ddh,[1 1 nClasses]);
      Dcl = Dch+repmat(Ddl,[1 1 nClasses]);
    end
  end
end
% now choose the patterns
indih = zeros(nClasses,length(relind));
lah = zeros(nClasses,length(relind));
laha = zeros(nClasses,length(relind));

%nComps,nClasses+optfunc);

for i = 1:nClasses
  DD = diag(Dch(:,:,i));
  [DD,I] = sort(DD);
  indih(i,:) = I(end:-1:1)';
  lah(i,:) = diag(Dc(indih(i,:),indih(i,:),i))';
  laha(i,:) = diag(Dch(indih(i,:),indih(i,:),i))';
%  for k = 1:optfunc
%      la(hi,nClasses+k) = diag(D(ind(hi,i),ind(hi,i),k+nClasses));
%    end
  if ~strcmp(choice,'s')
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
if strcmp(kernel,'linear') % simple case
  epo = proc_linearDerivation(epo,w);
else
  epo = proc_kernelProjection(epo,w,dat(:,:),kernel,varargin{:});
end


% cosmetic correction
epo.clab = cell(1,length(INDI));
for i = 1:length(INDI)
  [I,J] = find(INDI(i)==ind);
  epo.clab{i} = sprintf('Class %s, Pattern %i',epo.className{J(1)}, I(1));
  for j = 2:length(I)
    epo.clab{i} = sprintf('%s;Class %s, Pattern %i',epo.clab{i},epo.className{J(j)}, I(j));
  end
end



  
% subfunction kernels

function K = poly(dat1,dat2,p,c)
if ~exist('p','var') | isempty(p)
  p = 2;
end

if ~exist('c','var') | isempty(c);
  c = 0;
end

K = (dat1'*dat2+c).^p;



function K = gauss(dat1,dat2,sigma)
if ~exist('sigma','var');
  sigma = 1;
end

n1 = size(dat1,2);
n2 = size(dat2,2);

K = zeros(n1,n2);
if n1>n2
  for i = 1:n2
    da = dat1-repmat(dat2(:,i),[n1,1]);
    da = sum(da.*da,1);
    K(:,i) = exp(-0.5*da'/sigma);
  end
else
  for i = 1:n1
    da = dat2-repmat(dat1(:,i),[n2,1]);
    da = sum(da.*da,1);
    K(i,:) = exp(-0.5*da/sigma);
  end
end

