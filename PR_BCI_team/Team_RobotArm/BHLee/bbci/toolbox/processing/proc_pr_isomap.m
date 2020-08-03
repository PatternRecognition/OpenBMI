function [ fv, score ] = proc_pr_isomap(fv, d, varargin)
%PROC_PR_ISOMAP   Perform Isomap.
%
%usage
%  [ fv, score ] = proc_pr_isomap(fv, d, <options>)
%
%input
%  fv       Feature vectors.
%  d        Desired number of dimensions.
%  options  Options (see below).
%
%output
%  fv       Projected feature vectors.
%  score    Relevance score on each dimension.
%
%options
%  verbosity    If verbosity>1 additional information is written to the 
%               command line. Default: 0
%  nfcn         Neighborhood-function, 'knn' or 'eps'. Default: 'knn'.
%               'knn':
%                 Build graph using the k-nearest-neighbor rule. 
%                 See parameter k. 
%               'eps':
%                 Build graph using the epsilon-ball rule where two
%                 vertices are connected iff their distance is < eps.
%                 See parameter eps.
%  k            Parameter k for the k-nearest-neighbor rule. Default: 7.
%  eps          Parameter eps for the epsilon-ball rule. Default: 0
%  compi        If there are several connected components, this
%               option lets you choose the component (ordered by size, starting
%               with index 1 which is the largest one) that should be embedded. 
%               Default: 1 
%  D            Distance matrix. If not given, D is computed as euclidean
%               distance matrix from fv.x. Default: []
%
%note
%  When the graph is not connected the number of datapoints change. This
%  will most likely confuse the calling function. Be prepared.
%
%see also
%  J. B. Tenenbaum, V. de Silva, J. C. Langford (2000).  A global
%  geometric framework for nonlinear dimensionality reduction.  
%  Science 290 (5500): 2319-2323, 22 December 2000.  
%
%  Cox, Cox. Multimdimensional Scaling.
%
%author
%  Originally by Josh Tenenbaum (see isomap.stanford.edu), adopted by
%  paul@first.fhg.de to fit in the idabox.
%
%    BEGIN COPYRIGHT NOTICE
%
%    Isomap code -- (c) 1998-2000 Josh Tenenbaum
%
%    This code is provided as is, with no guarantees except that 
%    bugs are almost surely present.  Published reports of research 
%    using this code (or a modified version) should cite the 
%    article that describes the algorithm: 
%
%    Comments and bug reports are welcome.  Email to jbt@psych.stanford.edu. 
%    I would also appreciate hearing about how you used this code, 
%    improvements that you have made to it, or translations into other
%    languages.    
%
%    You are free to modify, extend or distribute this code, as long 
%    as this copyright notice is included whole and unchanged.  
%
%    END COPYRIGHT NOTICE

% Handle options.
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
		'verbosity', 0, ...
		'nfcn', 'knn', ...
		'k', 7, ...
		'eps', 0, ...
		'compi', 1, ...
		'D', [] ...
		);
		
[ rx N ] = size(fv.x);
%if rx>N, warning('Feature vectors have more columns that rows. Looks very suspicious.'); end

% Compute distance matrix if necessary. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(opt.D) 
	% Code taken from harmeli's distanz.m which was inspired by
	% Roland Bunschoten.
	xx = sum(fv.x .* fv.x,1); 
	xz = fv.x'*fv.x;
    D = sqrt(abs(repmat(xx',[1 N]) - 2*xz + repmat(xx,[N 1])));
else 
	D = opt.D;
end

% Build the graph. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
INF = 1000*max(max(D))*N;  % Effectively infinite distance.
switch opt.nfcn
	case 'knn'
		[tmp, ind] = sort(D); 
     	for i=1:N
         	D(i,ind((2+opt.k):end,i)) = INF; 
		end
					
	case 'eps'
		warning off    %% Next line causes an unnecessary warning, so turn it off
		D =  D./(D<=opt.eps); 
		D = min(D,INF); 
		warning on
end
D = min(D,D');    % Make sure distance matrix is symmetric.

% Finite entries in D now correspond to distances between neighboring points. 
% Infinite entries (really, equal to INF) in D now correspond to 
% non-neighoring points. 
%
% We use Floyd's algorithm, which produces the best performance in Matlab. 
% Dijkstra's algorithm is significantly more efficient for sparse graphs, 
% but requires for-loops that are very slow to run in Matlab.  A significantly 
% faster implementation of Isomap that calls a MEX file for Dijkstra's 
% algorithm can be found in isomap2.m (and the accompanying files
% dijkstra.c and dijkstra.dll). 

% Compute pairwise-distance matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic; 
for k=1:N
     D = min(D,repmat(D(:,k),[1 N])+repmat(D(k,:),[N 1])); 
     if ((opt.verbosity == 1) & (rem(k,20) == 0)) 
          disp([' Iteration: ' num2str(k) '     Estimated time to completion: ' num2str((N-k)*toc/k/60) ' minutes']); 
     end
end

% Select connected component %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_connect = sum(~(D==INF));        % Number of points each point connects to.
[tmp, firsts] = min(D==INF);       % First point each point connects to. 
                                   % I, Paul, would put it this way: First
								   % (in sense of indices) point
								   % that each point is not connected to.
[comps, I, J] = unique(firsts);    % Represent each connected component once.
size_comps = n_connect(comps);     % Size of each connected component.
[tmp, comp_order] = sort(size_comps);  % Sort connected components by size.
comps = comps(comp_order(end:-1:1));    
size_comps = size_comps(comp_order(end:-1:1)); 
n_comps = length(comps);               % Number of connected components.

if n_comps > 1, warning('Graph is not connected, embedding only one connected component.'); end

indices = find(firsts==comps(opt.compi)); 
D = D(indices, indices); 
N = length(indices); 

% Perform Classical MDS on the distance matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eigsopts.disp = opt.verbosity;
[vec, val] = eigs(-.5*(D.^2 - sum(D.^2)'*ones(1,N)/N - ones(N,1)*sum(D.^2)/N + sum(sum(D.^2))/(N^2)), d, 'LR', eigsopts); 

h = real(diag(val)); 
[foo,sorth] = sort(h);  sorth = sorth(end:-1:1); 
val = real(diag(val(sorth,sorth))); 
vec = vec(:,sorth); 

fv.x = real(vec(:,1:d).*(ones(N,1)*sqrt(val(1:d)')))'; 
score = h(1:d) ./ max(h(1:d));

