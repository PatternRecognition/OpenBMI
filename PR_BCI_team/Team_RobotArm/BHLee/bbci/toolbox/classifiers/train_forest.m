function C = train_forest(Xtr, Ytr, varargin)
%
% train_forest: train an ensemble of trees using either 
%     Breiman's random forest methodology or AdaBoost
%
% Synopsis:
%    forest = train_forest(Xtrain,Ytrain)
%    forest = train_forest(Xtrain,Ytrain,Property,Value,...)
%    
%  Arguments:
%    Xtrain: [d N] matrix of training data (N points in d dimensions)
%    Ytrain: [k N] matrix. Treated as classification if k>1, else
%                  as regression. 
%    
%  Returns:
%    forest: ensemble of tree structure
%
%
%  Properties:
%     'nb_trees'  : (integer) number of trees in the ensemble,
%                    default 50
%
%
%      'ensemble_type' : (string) type of ensemble methodology, in
%             'Adaboost': AdaBoost (classification only, multiclass OK)
% 	      'bagging' : Breiman's bagging
% 	      'random_nobagging' : Random subset selection of features
% 	                       at each node for training (default
% 	                       because it is my favorite)
% 	      'random'  : Breiman's random forest (combination of the
% 	                         two above)
%
%      'nb_sel_features' : (integer or double in [0..1]). Relevant
% 	       only for random methods. Indicate the number (if integer)
% 	       or proportion (if double in [0..1]) of the random features
% 	       picked at each node for selection during construction.
% 	       Default 0.1 . This is an important parameter. Using
% 	       coarse crossval on this is recommended.
%	 
%      'max_depth' : (integer) see train_tree.
%                    default: Inf for random ensembles
% 		            ceil(log10(N)) for adaboost
% 	       This can be important. Testing a few differerent values
% 	       with a validating set is recommended (in particular
%              if using AdaBoost)
%
%      'calibrate_var' : (bool) (regression only) first perform variance
%              prediction calibration (scaling factor) using a 25%
%              validation set. Therefore, double the computation time.
%              Only useful if variance prediction is wanted.
%              NB: does not seem to work very well actually. So try
%              it only if the results without variance calibration
%              are really horrible.
%			      
%      'min_points' : (integer) see train_tree (default 5)
%     
%      'pruning_parameter' : (double) see train_tree
% 	                     (default 0, no pruning). Having a
% 	                     coarse crossval on this is recommended
% 	                     (in particular for Adaboost)
%
%     'proba_output' : (boolean) relevant for classification
%              only. Ouput estimated probability vector over classes instead
%              of just classification. Default false.
% 
%     'tree_proba_output' : (boolean) request probabilistic output
%              for single trees. Leave it to default unless there is
%              a reason. Default is false for AdaBoosted forests,
%	       true for random forests.
%
%      'split_criterion' : (string) see train_tree
%     
%      'epsilon' : (double) precision value for some thresholds.
%        (default 1e-9)
%       
%   NB: 'pruning_parameter', 'min_points' and 'max_depth' are in
%   part redundant. I would recommend to perform cross-val only on
%   one of these paramters and leave the others at their default value.
%
%   References: Random Forests. L. Breiman, Machine Learning
%
%   See also : apply_forest, train_tree, apply_tree   
%

%  Author(s) : Gilles Blanchard 12/04/06
%

Ytr = double(Ytr);

error(nargchk(2, Inf, nargin));

opt = propertylist2struct(varargin{:});

[ndim, npoints] = size(Xtr);
[nclasses, checknpoints] = size(Ytr);


[opt, isdefault] = set_defaults(opt, 'pruning_parameter', 0, ...
				     'ensemble_type', 'random_nobagging', ...
				     'nb_sel_features', 0.1,...
				     'min_points', 5, ...
				     'epsilon', 1e-9, ...
				     'nb_trees', 50,...
				     'calibrate_var', false,...
				     'proba_output', false);


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

switch(lower(opt.ensemble_type))
 case {'random', 'random_nobagging'}
  [opt, isdefault] = set_defaults(opt, 'random_tree', true, ...
				       'max_depth', Inf,...
				       'tree_proba_output', true);
 case 'bagging'
  [opt, isdefault] = set_defaults(opt, 'random_tree', false, ...
				       'max_depth', Inf,...
				       'tree_proba_output', true);
 case 'adaboost'
  if (~strcmp(opt.problem_type, 'classification') )
    error('Adaboost only implemented for classification')
  end;
  
  [opt, isdefault] = set_defaults(opt, 'random_tree', false, ...
				       'max_depth', ceil(log10(npoints)),...
				       'tree_proba_output', false);
  [maxfoo, Yclass] = max(Ytr,[],1);
 otherwise
  error('Unknown ensemble type');
end;

C{1}.ensemble_type = opt.ensemble_type;
C{1}.proba_output = opt.proba_output;
C{1}.variance_correct_factor = 1;

if (strcmp(lower(opt.problem_type),'regression') & opt.calibrate_var)
  % variance calibration
  opt.calibrate_var = false;
  varlist = [ fieldnames(opt)' ; struct2cell(opt)'];
  disp('Calibrating variance prediction.');
  shuffle = randperm(npoints);
  subtrain = shuffle(1:floor(npoints*3/4));
  subval = shuffle((floor(npoints*3/4)+1):end);
  subforest = train_forest(Xtr(:,subtrain), Ytr(:,subtrain), varlist{:});
  [out, predvar] = apply_forest(subforest, Xtr(:,subval));
  C{1}.variance_correct_factor = mean( (Ytr(:,subval)-out).^2./predvar);
  clear shuffle subval subtrain subforest;
  disp('Finished calibration.');
end;

varlist = [ fieldnames(opt)' ; struct2cell(opt)'];

for ii = 1:opt.nb_trees
  
%  disp(ii);
  
  switch(lower(opt.ensemble_type))
    
   case {'random', 'bagging'}
    bagging_indices = ceil(rand(1,npoints)*npoints);
    Xbag = Xtr(:,bagging_indices);
    Ybag = Ytr(:,bagging_indices);
    C{ii}.weight = 1/opt.nb_trees;
    C{ii}.tree = train_tree(Xbag, Ybag, varlist{:});

   case 'random_nobagging'
    C{ii}.weight = 1/opt.nb_trees;
    C{ii}.tree = train_tree(Xtr, Ytr, varlist{:});
    
   case 'adaboost'
    C{ii}.tree = train_tree(Xtr, Ytr, varlist{:});
    out = apply_tree(C{ii}.tree, Xtr, 'proba_output', false);
    
    [maxfoo, outclass] = max(out,[],1);
    Yweights = sum(Ytr,1)/sum(sum(Ytr));
    
    errors = abs(outclass-Yclass)>opt.epsilon;
    
    werror = sum(errors.*Yweights);
    
    C{ii}.weight = log( (1-werror)/werror*(nclasses-1));


    Ytr(:,errors) = Ytr(:,errors)*(1-werror)/werror*(nclasses-1);


    Ytr = Ytr/sum(sum(Ytr));
    
  end;

  C{ii}.tree{1}.proba_output = opt.tree_proba_output;
  
end;


%if (strcmp(lower(opt.problem_type),'regression'))   % This is a
%                                                    % poor hack
%  C{1}.variance_correct_factor = 1;
%  [out, predvar] = apply_forest(C, Xtr);
%  C{1}.variance_correct_factor = mean( (Ytr-out).^2./predvar);
%end;