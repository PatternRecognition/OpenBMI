function C = train_tree(Xtr, Ytr, varargin)
% 
% train_tree : housemade implementation of some standard tree methods
%
%
% Synopsis:
%    tree = train_tree(Xtrain,Ytrain)
%    tree = train_tree(Xtrain,Ytrain,Property,Value,...)
%    
%  Arguments:
%    Xtrain: [d N] matrix of training data (N points in d dimensions)
%    Ytrain: [k N] matrix. Treated as classification if k>1, else
%                  as regression. 
%                  For classification, points can be
%                  weighted, i.e. the class index can belong to [0..1]
%    
%  Returns:
%    tree: trained tree structure
%
%
%  Properties:
%
%     'max_depth' : (integer) maximum depth of tree (default Inf)
%    
%     'min_points' : (integer) minimal number of points in a node to allow
% 	splitting (default 5)
%	
%     'split_criterion' : (string) criterion used for splitting nodes.
%       For classification: in { 'Gini' [default], 'entropy', 'twoing'}.
%       For regression:     in { 'leastsquares' [default], 'lognormal'}.      
%      
%     'pruning_parameter' : (double) multiplier of the size penalization
%         for pruning. 
% 	0: no pruning (default)
%          If you use pruning, for classification try e.g. the range [0.1 10]
%             for regression if you have an idea of the typical
%             noise variance try the range 
%              [0.05 5]*(estimated noise variance)
%          NB: when using forests always include the value 0 (no
%          pruning) in your search range.
%
%     'random_tree' : (boolean) use random tree building procedure. Makes sense
% 	only if using a random forest. See train_forest.
%
%     'proba_output' : (boolean) relevant for classification
%      only. Ouput estimated probability vector over classes instead
%      of just classification. Default false.
%
%     'nb_sel_features' : parameter only meaningful for random tree.
%       See train_forest for more details. Default 0.1
%      
%     'epsilon' : (double) precision for some thresholds. Default 1e-9.
%    
%
%   References: Classification and Regression Trees. Breiman, Friedman, Olshen,
%       Stone. Belmont, CA: Wadsworth International Group.
%
%   See also : apply_tree, train_forest, apply_forest   
%

%  Author(s) : Gilles Blanchard 12/04/06
%


error(nargchk(2, Inf, nargin));

opt = propertylist2struct(varargin{:});

[opt, isdefault] = set_defaults(opt, 'pruning_parameter', 0, ...
				     'random_tree', false, ...
				     'nb_sel_features', 0.1,...
				     'min_points', 5, ...
				     'epsilon', 1e-9, ...
				     'max_depth', Inf,...
				     'proba_output', false);


[ndim, npoints] = size(Xtr);
[nclasses, checknpoints] = size(Ytr);

if (nclasses == 1)
  opt.problem_type = 'regression';
else
  opt.problem_type = 'classification';
end;

switch(lower(opt.problem_type))
 case 'classification'
  [opt, isdefault] = set_defaults(opt, 'split_criterion', 'Gini');
 case 'regression'
  [opt, isdefault] = set_defaults(opt, 'split_criterion', 'leastsquares');
 otherwise
  error('Unknown problem type!');
end;


if (opt.nb_sel_features<1)
  opt.nb_sel_features = ceil(opt.nb_sel_features*ndim);  % percentage of features
end;

% add some error checking here

nodesamples{1} = 1:npoints; %samples at each node, kept in a
                            %separate array because we won't need
                            %them at the end
			    
tree{1}.depth = 1;
tree{1}.parent = 0;
tree{1}.tree_type = opt.problem_type;
tree{1}.proba_output = opt.proba_output;
tree{1}.type = 'final';
tree{1}.opt = opt;


if (strcmp(opt.problem_type,'classification'))
  tree{1}.nb_classes = nclasses;
end;

index = 1; %index of current treated node
maxnodes = 1; %current number of nodes in structure

while (index <= maxnodes)       % build the tree


%  disp([min(nodesamples{index})   max(nodesamples{index}) size(nodesamples{index})]);
  
  
  switch(lower(opt.problem_type))
   
   case 'classification'

    class_counts = sum( Ytr(:,nodesamples{index}) , 2);
    tree{index}.class_probas = class_counts/sum(class_counts);
    [foo tree{index}.class] = max(tree{index}.class_probas);
    tree{index}.error = sum(class_counts)-class_counts(tree{index}.class);
    cur_error = tree{index}.error;
    tree{index}.totalpoints = sum(class_counts);
   
   case 'regression'
    
    tree{index}.mean = mean( Ytr(1,nodesamples{index}) );
    tree{index}.var  = var( Ytr(1,nodesamples{index}) );
    tree{index}.totalpoints = length(nodesamples{index});
    cur_error = tree{index}.var*tree{index}.totalpoints;
   otherwise
    error('Unknown problem type');
  
  end;
    
    
    
  if ( (length(nodesamples{index}) >= opt.min_points)...  % try to cut
       && (tree{index}.depth < opt.max_depth ))         % only if
							% enough points and max_depth not reached
							
    if (opt.random_tree)
      features = 1:ndim;

      for ii = 1:opt.nb_sel_features
	findex = ii + floor(rand*(ndim+1-ii));
	if (findex > ndim)
	  disp('error');
	end;	  
	switcho = features(findex);
%	features(index) = features(ii);   % again a bug was hiding here
	features(findex) = features(ii);
	features(ii) = switcho;
      end;
      sel_features = features(1:opt.nb_sel_features);

    else
      sel_features = 1:ndim;
    end;
    
    best_crit = Inf;
    best_feature = 0;
    best_threshold = -Inf;
      

    
    for ii = sel_features    %compute best feature/threshold
      [point_values, point_order] = sort(Xtr(ii, nodesamples{index}));
      
      nbpin = length(nodesamples{index});
      
      thresholds = (point_values(1:(nbpin-1)) + point_values(2:nbpin))/2;
                             %  consider midpoints as thresholds
      point_diff = point_values(1:(nbpin-1)) - point_values(2:nbpin);
	         % to detect repeated values later
		 
		 
      switch (lower(opt.problem_type))
       case 'classification'
	class_counts_right = class_counts;
	class_counts_left = zeros(nclasses,1);
	
	for t = 1:length(thresholds) 
	  class_counts_right = class_counts_right - ...
	      Ytr(:, nodesamples{index}(point_order(t)));
	  class_counts_left = class_counts_left + ...
	      Ytr(:, nodesamples{index}(point_order(t)));
	  if (point_diff(t) < - opt.epsilon)
	    crit = eval_crit(opt.split_criterion, class_counts_right,...
			     class_counts_left);
	    if (crit < best_crit)
	      best_crit = crit;
	      best_feature = ii;
	      best_threshold = thresholds(t);
	    end;
	  end;
	end;
	
       case 'regression'
%	Yright = Ytr(1, point_order(end:-1:1)); % Huge bug previously here??

	Ynode = Ytr(1, nodesamples{index}(point_order));

	nleft = 0;
	nright = nbpin;
	sp_index=1;
	Ysum_right = sum(Ynode);
	Ysum_left = 0;

	if (nbpin <1)
	  disp('Error?');
	end;
	
	for t = 1:length(thresholds) 

	  switch (lower(opt.split_criterion))
	   case 'leastsquares'
	    nleft = nleft+1;
	    nright = nright-1;
	    swpoint = Ynode(t);
	    Ysum_right = Ysum_right - swpoint;
	    Ysum_left = Ysum_left + swpoint;
	   case 'lognormal'
	    sp_index=sp_index+1;
	  end;

	  if (point_diff(t) < - opt.epsilon)
	    switch (lower(opt.split_criterion))
	     case 'leastsquares'
	      crit = - Ysum_right*Ysum_right/nright ...
	      - Ysum_left*Ysum_left/nleft;
	     case 'lognormal'
	      sigmaleft = var(Ynode(1:(sp_index),1));
	      sigmaright = var(Ynode((sp_index+1):end,1));
	      crit = -sp_index*log(sigmaleft) - (nbpin-sp_index)*log(sigmaright);
	     otherwise
	      opt.split_criterion
	      error('Unknown splitting criterion');
	    end;

	    if (crit < best_crit)
	      best_crit = crit;
	      best_feature = ii;
	      best_threshold = thresholds(t);
	    end;
	  end;
	end;
      end;
    end;

    if ( best_feature > 0 && cur_error > 0 ) %cut only if improvement
      
      tree{index}.type = 'split';
      tree{index}.cut_dim = best_feature;
      tree{index}.cut_value = best_threshold;

      leftsamples = nodesamples{index}(Xtr(best_feature, ...
					   nodesamples{index}) < best_threshold); 
      rightsamples = nodesamples{index}(Xtr(best_feature, ...
			    nodesamples{index}) >= best_threshold);
      tree{index}.leftchild = maxnodes + 1;
      tree{index}.rightchild = maxnodes + 2;
      nodesamples{maxnodes+1} = leftsamples;
      nodesamples{maxnodes+2} = rightsamples;
      tree{maxnodes+1}.depth = tree{index}.depth + 1;
      tree{maxnodes+2}.depth = tree{index}.depth + 1;
      tree{maxnodes+1}.parent = index;
      tree{maxnodes+2}.parent = index;
      tree{maxnodes+1}.type = 'final';
      tree{maxnodes+2}.type = 'final';
      maxnodes = maxnodes + 2;
    end;
  end;
    
  index = index + 1;
  
end;
	    
if (opt.pruning_parameter > opt.epsilon)
  tree = prune_tree(tree,opt.pruning_parameter);
end;

C = tree;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function crit = eval_crit(criterion, class_counts_right, class_counts_left)

if (nargin==3)
  pleft = sum(class_counts_left);
  if (pleft > 0)
    probaleft = class_counts_left/pleft;
  else
    probaleft = 0;
  end;
else
  pleft = 0;
  probaleft = 0;
end;

pright = sum(class_counts_right);

if (pleft > 0)
  probaright = class_counts_right/pright;
else
  probaright = 0;
end;



ptot = pright+pleft;

pleft = pleft/ptot;
pright = pright/ptot;



switch (lower(criterion))
 case 'gini'
  
  crit =  - pleft*sum(probaleft.^2) - pright*sum(probaright.^2);
 
 case 'entropy'

 % S = warning( 'off', 'MATLAB:logOfZero'); % This does not work
                                            % (Matlab bug??)

  probaleft(probaleft < 1e-30) = 1e-30;
  lprobleft = probaleft.*log(probaleft);

  probaright(probaright < 1e-30) = 1e-30;
  lprobright = probaright.*log(probaright);


  crit = -pleft*sum(lprobleft) - pright*sum(lprobright);
  
%  warning(S);
  
 case 'twoing'
  
  crit = - pleft*pright*sum(abs(probaleft-probaright))^2;
  
 otherwise

  error('Unknown splitting criterion');

end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function treeout = prune_tree(treein, pruning_parameter)

treesize = length(treein);

switch (lower(treein{1}.opt.problem_type))
 case 'classification'
  for i=1:treesize  % I still don't know how to do this without an explicit loop
    errorlist(i) = treein{i}.error + pruning_parameter;
  end;
 case 'regression'
  for i=1:treesize
    errorlist(i) = treein{i}.var*treein{i}.totalpoints + pruning_parameter;
  end;
 otherwise
  error('Unknown problem type');
end;

besterrorlist = errorlist;

for indexx = treesize:-1:1;
  switch (treein{indexx}.type)
   case 'split'
    curbesterror = besterrorlist(treein{indexx}.leftchild) + ...
	besterrorlist(treein{indexx}.rightchild);
    if (errorlist(indexx) < curbesterror)
      treein{indexx}.type = 'final';
    else
      besterrorlist(indexx) = curbesterror;
    end;
  end;
end;

%% now compactify tree by taking out cut nodes

indexin = 1; %index of current treated node
indexout = 1; 
maxnodesout = 1; %current number of nodes in out tree
parentarray = 0;

treeout{indexout} = treein{indexin};
switch( treein{indexin}.type )
 case 'split'
  treeout{indexout}.leftchild = maxnodesout + 1;
  treeout{indexout}.rightchild = maxnodesout + 2;
  parentarray(maxnodesout+1) = indexout;
  parentarray(maxnodesout+2) = indexout;
  maxnodesout = maxnodesout + 2;
 case 'final'
  if isfield(treeout{indexout},'richtchild')
    treeout{indexout} = rmfield(treeout{indexout},{'leftchild' ...
		    'rightchild'});
  end;
end;
indexin = indexin+1;
indexout = indexout+1;


while (indexout <= maxnodesout)       % build the new tree
  switch (treein{treein{indexin}.parent}.type) % test if parent not cut
   case  'split'
    treeout{indexout} = treein{indexin};
    treeout{indexout}.parent = parentarray(indexout);
    switch( treein{indexin}.type )
     case 'split'
      treeout{indexout}.leftchild = maxnodesout + 1;
      treeout{indexout}.rightchild = maxnodesout + 2;
      parentarray(maxnodesout+1) = indexout;
      parentarray(maxnodesout+2) = indexout;
      maxnodesout = maxnodesout + 2;
     case 'final'
      if isfield(treeout{indexout},'richtchild')
	treeout{indexout} = rmfield(treeout{indexout},{'leftchild' ...
		    'rightchild'});
      end;
    end;
    indexout = indexout+1;
   case 'final'
    treein{indexin}.type = 'final'; %propagate cut state
  end;
  indexin = indexin+1;
end;