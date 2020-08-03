function [fv,csp_w,csp_a,csp_r,theta] = proc_csp_prototypes(fv,nCSP,nIBICA,varargin)
% [fv,csp_w] = proc_csp_prototypes(fv,nCSP,nIBICA,<opt>)
% 
% find the prototypical CSP patterns in old sessions.
% fv must contain a field '.session'.

% kraulem 06/06
if nargin>3
  opt = propertylist2struct(varargin{:});
else
  opt = struct;
end
opt = set_defaults(opt,'type','filters',...
		       'cov','average',...
		       'method','gamma');

% first, generate CSP filters and patterns:
sessions = unique(fv.session);
csp_w_old = [];
csp_r_old = [];
csp_a_old = [];
csp_r = [];
csp_w = [];
csp_a = [];
for ii = 1:length(sessions)
  % for each experiment: find the first nCSP patterns.
  fv_tmp = proc_selectEpochs(fv,find(fv.session==sessions(ii)));
  Sig_sqrt = sqrtm(2*proc_get_covariances(fv_tmp,'classwise',2));
  [fv_tmp,csp_w,la,csp_a] = proc_csp3(fv_tmp,nCSP);
  csp_w_old = [csp_w(:,1:nCSP) csp_w_old csp_w(:,(nCSP+1):end)];
  csp_r_old = [Sig_sqrt*csp_w(:,1:nCSP) csp_r_old Sig_sqrt*(csp_w(:,(nCSP+1):end))];
  csp_a_old = [csp_a(1:nCSP,:)' csp_a_old csp_a((nCSP+1):end,:)'];
end
switch opt.type
 case 'filters'
  % use the CSP filters for the distance metric.
  % heuristic: try to mirror the patterns to one hemisphere.
  %csp_old = mirror_cluster(csp_old);
  [w1,theta1,d,d,ind1] = ibica2(csp_w_old(:,1:length(sessions)*nCSP),[],nIBICA,5,0,opt.method,0);
  [w2,theta2,d,d,ind2] = ibica2(csp_w_old(:,(length(sessions)*nCSP+1):end),[],nIBICA,5,0,opt.method,0);
  csp_w = [w1,w2];
  theta = [theta1;theta2];
  csp_r = csp_r_old(:,[ind1,ind2+length(sessions)*nCSP]);
  csp_a = csp_a_old(:,[ind1,ind2+length(sessions)*nCSP]);
  %csp_w = ibica2(csp_old,[],nIBICA,5,0,[],0);
 case 'patterns'
  % Use CSP patterns instead of filters and generate filters from the most stable ones.
  csp_a = [ibica2(csp_a_old(:,1:length(sessions)*nCSP),[],nIBICA,5,0,opt.method,0),...
	   ibica2(csp_a_old(:,(length(sessions)*nCSP+1):end),[],nIBICA,5,0,opt.method,0)];

  % calculate the filter from the patterns.
  csp_w = get_filter(fv,csp_a,csp_a_old,csp_w_old,opt.cov);
 case 'rotations'
  % Use CSP rotations instead of filters and generate filters from the most stable ones.
  csp_r = [ibica2(csp_r_old(:,1:length(sessions)*nCSP),[],nIBICA,5,0,opt.method,0),...
	   ibica2(csp_r_old(:,(length(sessions)*nCSP+1):end),[],nIBICA,5,0,opt.method,0)];
  csp_w = get_filter(fv,csp_r,csp_r_old,csp_w_old,opt.cov);
end
fv = proc_linearDerivation(fv,csp_w);
return
function csp_w = get_filter(fv,csp_a,csp_a_old,csp_w_old,cov_type)
% get the covariance matrix correctly
if strcmp(cov_type,'average')
  [cov_mat,n,className] = proc_get_covariances(fv,'classwise',2);
  cov_inv = inv(cov_mat);
  csp_w = cov_inv*csp_a*inv(csp_a'*cov_inv*csp_a);
elseif strcmp(cov_type,'average_rotation')
  [cov_mat,n,className] = proc_get_covariances(fv,'classwise',2);
  cov_inv = inv(sqrtm(cov_mat));
  csp_w = cov_inv*csp_a;
elseif strcmp(cov_type,'match')
  % get the matching filters
  
  % normalize csp_a_old:
  csp_a_old = csp_a_old./repmat(sqrt(sum(csp_a_old.^2,1)),size(csp_a_old,1),1);
  for ii = 1:size(csp_a,2)
    % for each pattern: find the original pattern in the matrix.
    d_mat = csp_a_old-repmat(csp_a(:,ii),1,size(csp_a_old,2));
    [d,ind] = min(sum(d_mat.^2,1));
    csp_w(:,ii) = csp_w_old(:,ind);
  end
elseif isnumeric(cov_type)
  % cov_type is a covariance matrix of some epoched data.
  % JEZ Formula:
  %cov_inv = inv(cov_type);
  %csp_w = cov_inv*csp_a*inv(csp_a'*cov_inv*csp_a);
  % Ryota Formula (only makes sense with rotations):
  csp_w = inv(sqrtm(cov_type))*csp_a;
else
  error('Other covariance types not implemented yet.');
end
return





function [A,theta,cx,lpart,ind] = ibica2(x,lpart,nsrc,k,thr,method,verbose)
%IBICA2 does inlier-based ICA.
%
% usage:
%    [A,theta,cx,lpart,ind] = ibica2(x,lpart,nsrc,k,thr,method,verbose);
%    or simply
%    A = ibica2(x);
%
% inputs:
%    x        data with the signals along the rows
%    lpart    the size of each partition, default = min(1000,cx)
%    nsrc     desired number of sources; if nsrc == -1, then ibica2
%             choose automagically the number of sources,
%             default = size(x,1)
%    k        for gamma index (k nearest neighbors),
%             default = floor(lpart*0.05)
%    thr      what percentage from the center is ignored, default = 0.2
%    method   which index to use, 'kappa','gamma','delta', default = 'gamma'
%    verbose  shall the algorithm print infos? default = 1 (yes)
%
% outputs:
%    A        the estimated mixing matrix
%    theta    the winning indices
%    cx       how many data points have been used, the number of data points
%             can be reduced by the partitioning
%    lpart    the actual partition size, see cx
%    ind_x    the positions of the winners in x.
% sth/fcm * 19aug2004

[rx,cx] = size(x);

if ~exist('lpart','var')|isempty(lpart), lpart = min(1000,cx); end;
if ~exist('nsrc')|isempty(nsrc), nsrc = rx; end
if ~exist('k')|isempty(k), k = floor(lpart*0.05); end;
if ~exist('thr')|isempty(thr), thr = 0.2; end;
if ~exist('method')|isempty(method), method = 'gamma'; end
if ~exist('verbose')|isempty(verbose), verbose = 1; end

% shall we choose the number of sources automagically?
if nsrc == -1
  error('not yet implemented')
  nsrc = cx;
  autonsrc = 1;
else
  autonsrc = 0;
end

% remove points from the inner sphere
xn = sqrt(sum(x.*x,1));  % the norms of the columns of x
[dummy,idx] = sort(-xn);
% pick (1-thr)-many the points with largest norm and don't change their ordering
x = x(:,sort(idx(1:floor(cx*(1-thr)))));
cx = size(x,2);
clear dummy idx xn

% project data sets to unit sphere
x = x./repmat(sqrt(sum(x.*x,1)),rx,1);

% partition the data into npart sets
lpart = min(lpart,cx);    % update lpart just in case
npart = round(cx/lpart);  % how many partitions of approximately lpart size
lpart = floor(cx/npart);  % the actual partition size
parts = randperm(cx);
parts = parts(1:(npart*lpart));
% randomly cut off the points not needed
x = x(:,parts);
% get the new length of x, should be npart*lpart
cx = size(x,2);
% note that the order of x changed
% the partitions are  x(:,1:lpart), x(:,lpart+(1:lpart)), ...
clear parts
if verbose
    fprintf([mfilename ' divided the data into %d partitions of size %d.\n'],npart,lpart)
end

% calculate the initial indices
idx = zeros(cx,1);
ind_x = [];
for i=1:npart 
  % calculate all distances per partition
  ind = lpart*(i-1) + (1:lpart);
  xx = sum(x(:,ind).*x(:,ind),1);
  xz = x(:,ind)'*x(:,ind);
  d1 = abs(repmat(xx',[1 lpart]) - 2*xz + repmat(xx,[lpart 1]));
  d2 = abs(repmat(xx',[1 lpart]) + 2*xz + repmat(xx,[lpart 1]));
  d = sqrt(min(d1,d2));
  
  % sort the distances
  [dummy,r] = sort(d,1);
  for j = 1:lpart
    jj = ind(j);
    switch method % the outlier indices
     case 'delta'
      del1 = repmat(x(:,jj),[1,k]) ...
	     - x(:,ind(r(2:k+1,j)));
      del2 = repmat(x(:,jj),[1,k]) ...
	     + x(:,ind(r(2:k+1,j)));
      ndel1 = ...
	  sum(del1.*del1,1); ...
      % squared norms of del1
      ndel2 = ...
	  sum(del2.*del1,2);  % squared norms of del2
      del = mean([del1(ndel1<=ndel2),del2(ndel1>ndel2)],2);
      idx(jj) = sqrt(del'*del);
     case 'gamma'
      idx(jj) = mean(d(j,r(2:k+1,j)));
     case 'gamma_median'
      idx(jj) = median(d(j,r(2:k+1,j)));
     case 'kappa'
      idx(jj) = d(j,r(k+1,j));
     otherwise
      error('unknown method')
    end
  end
end

% We can not find more sources than data points
nsrc = min(nsrc,cx);
theta = zeros(nsrc,1);

% iteratively pick the columns of the mixing matrix
A = zeros(rx,nsrc);
kgds = idx;
mins = inf*ones(cx,1);
src = 1;
xx = sum(x.*x,1);
return_theta = zeros(nsrc,1);
while 1  % REPEAT until
  if verbose, fprintf('\r%d/%d',src,nsrc); end
  
  % choose the next column
  [theta(src),colidx] = min(idx);
  A(:,src) = x(:,colidx);
  idx(src) = colidx;
  return_theta(src) = kgds(colidx);
  src = src + 1;

  % repeat UNTIL
  if src > nsrc, break, end
  
  % calculate all distances to
  yy = sum(x(:,colidx).*x(:,colidx),1);
  xy = x'*x(:,colidx);
  d1 = abs(xx' + repmat(yy,[cx 1]) - 2*xy);
  d2 = abs(xx' + repmat(yy,[cx 1]) + 2*xy);
  d  = sqrt(min(d1,d2));
  
  % update the indices
  mins = min(mins,d);
  idx = inf*ones(cx,1);   % avoid division by zero
  nonull = find(mins>0);
  idx(nonull) = kgds(nonull) ./ mins(nonull);
end
if verbose, fprintf('\n'); end

%%% Return the unscaled version of the indices!
theta = return_theta;

% choose the right number of sources automagically?
if autonsrc
  error('not yet implemented')
  % we exclude nsrc = 1;
  [dummy,nsrc] = max(theta(3:end)./theta(2:end-1));
  nsrc = nsrc + 1;
  A = A(:,1:nsrc);
  if verbose
    fprintf([mfilename ' chose automagically the number of sources to be %d.\n'],nsrc)
  end
end
