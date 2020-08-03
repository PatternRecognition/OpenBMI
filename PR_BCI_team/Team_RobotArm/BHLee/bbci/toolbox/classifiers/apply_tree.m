function [out, predvar] = apply_tree(C, testdata)
% [out, predvar] = apply_tree(C, testdata)
%
% apply_tree - apply single classification/regression tree
%
% Synopsis:
%   class = apply_tree(C, X, [Property, Value, ...])
%   [Ymean, predvar] = apply_tree(C, X, [Property, Value, ...])
%   
% Arguments:
%  C: Struct. Trained tree model, as output by
%      train_tree. 
%
%  X: [d N] matrix. Data to evaluate the tree model on (N
%      points in d dimensions)
%   
% Returns:
%
%  class: (classification case) [k N] matrix of classification in k
%  classes, or of probability vectors over classes (see Properties below)
%
%  Ymean: (regression case) predictive mean for each point
%  predvar: (regression case only) predictive variance for each test point
%   
% Description:
%   Evaluate a tree model on test data.
%
%
%   See also : train_tree, train_forest, apply_forest   
%

%  Author(s) : Gilles Blanchard 14/03/06
%

error(nargchk(2, Inf, nargin));

[ndim, npoints] = size(testdata);

stack(1) = 1;

nodesamples{1} = 1:npoints;
index = 0;

switch (lower(C{1}.tree_type))
 case 'classification'
  if (nargout > 1)
    error('Incorrect number of out arguments');
  end;
  out = zeros(C{1}.nb_classes, npoints);
 case 'regression'
  out = zeros(1, npoints);
  var = out;
 otherwise
  error('Problem in tree structure (wrong tree type)');
end;

while( index < length(stack) )
  
  index = index + 1;
  
  switch (lower(C{index}.type))
   case 'split'      % compute split test samples
    leftsamples = nodesamples{index}(testdata(C{index}.cut_dim, ...
		            nodesamples{index}) < C{index}.cut_value);
    rightsamples = nodesamples{index}(testdata(C{index}.cut_dim, ...
		          nodesamples{index}) >= C{index}.cut_value);
    nodesamples{C{index}.leftchild} = leftsamples;
    nodesamples{C{index}.rightchild} = rightsamples;
    stack = [stack C{index}.leftchild C{index}.rightchild];
   
   case 'final'     % compute outputs for samples in that node
    switch (lower(C{1}.tree_type))
      case 'classification'
       if (C{1}.proba_output)
	 
	 out(:,nodesamples{index}) = ...
	     repmat(C{index}.class_probas, [1 ...
		    length(nodesamples{index})]);
       else
	 out(C{index}.class, nodesamples{index}) = 1;
       end;
     case 'regression'
      out(:,nodesamples{index}) = C{index}.mean;
      predvar(:,nodesamples{index}) = C{index}.var;
    end;
   otherwise
    error('Problem within tree structure: wrong node type');
  end;
end;

%% dirty hack, only provisory

if (size(out,1)==2)
  out = out(2,:);
end;