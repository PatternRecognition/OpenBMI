function [ fv ] = proc_pr_lle(fv, d, varargin)
%PROC_PR_LLE  Locally linear projection.
%
%usage
%  [ fv ] = proc_pr_lle(fv, d, <options>)
%
%input
%  fv        Feature vectors.
%  d         Desired number of dimensions.
%  options   Options (see below).
%
%output
%  fv        Projected feature vectors.
%
%options
%  verbosity   If verbosity>1, then additional information is written to
%              the command line. Default: 0
%  k           Number of neighbors. Default: 5
%
%see also
%  Nonlinear dimensionality reduction by locally linear embedding.
%  Sam Roweis & Lawrence Saul.
%  Science, v.290 no.5500 , Dec.22, 2000. pp.2323--2326.
%
%author
%  Code taken from the lle homepage. Adopted by paul@first.fhg.de
%  to fit in the idabox.

% Handle options.
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
		'verbosity', 0, ...
		'k', 5 ...
		);
	
[D,N] = size(fv.x);
%if D>N, warning('Feature vectors have more columns than rows. Looks very suspicious.'); end

% Compute distance matrix and neighborships. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X2 = sum(fv.x.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*fv.x'*fv.x;
[sorted,index] = sort(distance);
neighborhood = index(2:(1+opt.k),:);

% Compute Gram-Matrix of quadratic form. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(opt.k>D) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end

W = zeros(opt.k,N);
for ii=1:N
   z = fv.x(:,neighborhood(:,ii))-repmat(fv.x(:,ii),1,opt.k); % shift ith pt to origin
   C = z'*z;                                        % local covariance
   C = C + eye(opt.k,opt.k)*tol*trace(C);                   % regularlization (K>D)
   W(:,ii) = C\ones(opt.k,1);                           % solve Cw=1
   W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
end;

M = sparse(1:N,1:N,ones(1,N),N,N,4*opt.k*N); 
for ii=1:N
   w = W(:,ii);
   jj = neighborhood(:,ii);
   M(ii,jj) = M(ii,jj) - w';
   M(jj,ii) = M(jj,ii) - w;
   M(jj,jj) = M(jj,jj) + w*w';
end;

% Solve eigenproblem. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if opt.verbosity > 0 
	options.disp = 1; 
else
	options.disp = 0;
end
options.isreal = 1; options.issym = 1; 
[Y,eigenvals] = eigs(M,d+1,0,options);

[foo, ii] = sort(-diag(eigenvals));
fv.x = Y(:,ii(2:d+1))'*sqrt(N); % bottom evect is [1,1,1,1...] with eval 0
