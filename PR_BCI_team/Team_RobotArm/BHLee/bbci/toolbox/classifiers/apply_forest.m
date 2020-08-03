function [out, predvar] = apply_forest(C, testdata)
% apply_forest - apply ensemble of trees model
%
% Synopsis:
%   class = apply_forest(C, X)
%   [Ymean, predvar] = apply_forest(C, X)
%   
% Arguments:
%  C: Struct. Trained forest model, as output by
%      train_forest. 
%
%  X: [d N] matrix. Data to evaluate the forest model on (N
%      points in d dimensions)
%   
% Returns:
%
%  class: (classification case) [k N] matrix of classification in k
%  classes, or of probability vectors over classes (depending on
%  options entered for training C)
%
%  Ymean: (regression case) predictive mean for each point
%  predvar: (regression case only) predictive variance for each test point
%   
% Description:
%   Evaluate a tree model on test data.
%
%
%  Properties:
%
%    
%   See also :  train_forest, train_tree, apply_tree   
%

%  Author(s) : Gilles Blanchard 14/03/06
%

error(nargchk(2, 2, nargin));

switch (lower(C{1}.tree{1}.tree_type))
 case 'classification'
  if (nargout > 1)
    error('Incorrect number of out arguments');
  end;
end;




[ndim, npoints] = size(testdata);

out = 0;
predvar = 0;
totalweight = 0;

for ii = 1:length(C)
  switch (lower(C{1}.tree{1}.tree_type))
   case 'classification'
    out = out + C{ii}.weight*apply_tree(C{ii}.tree, testdata );
    totalweight = totalweight + C{ii}.weight;
   case 'regression'
    [ cur_out(ii,:) , cur_predvar(ii,:)] = apply_tree(C{ii}.tree, testdata);
    cweights(ii) = C{ii}.weight;
  end;
end;



switch (lower(C{1}.tree{1}.tree_type))
 case 'classification'
  out = out/totalweight;
  if (~C{1}.proba_output)
    [maxfoo, classes]  = max(out,[],1);
    out = zeros(size(out));
    
    for ii = 1:npoints
      out(classes(ii),ii) = 1;
    end;
  end;
 case 'regression'
  out = cweights*cur_out/sum(cweights);

    
  predvar = (cweights*cur_predvar/sum(cweights) + var(cur_out, cweights))/2;
%  predvar.varmean = var(cur_out, cweights);
  
%  predvar.meanvar = cweights*cur_predvar/sum(cweights);

  predvar = predvar/C{1}.variance_correct_factor;
 % predvar = predvar/2;
							      
end;