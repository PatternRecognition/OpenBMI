function out= apply_baggedKNN(C, y)
% function out = apply_baggedKNN(C, y)
%
% Apply the bagged knn classifier.

nTrain= size(C.xTr, 2);
N= size(y, 2);
e = exist('distC');
if (e==2) | (e==3),
  % Very fast distance calculation in C using atlas/blas
  % To be found in $NEURO_TOOL/matlab/mexfiles/distC/
  dist = distC( C.xTr, y );
else
  warning('Using old, very slow distance calculation.');
  dist= zeros(nTrain, N);
  % This is terribly slow. See below for proper replacement.
  for n= 1:N,
    for j= 1:nTrain,
      dist(j,n)= norm( C.xTr(:,j) - y(:,n) );
    end
  end 
end

[so,si]= sort(dist);

% Bootstrap replicates
for rep=1:C.B
  % nTrain-times draw a random number in 1-nTrain
  b_idx = sort(round( 1 + (nTrain-1) * rand(1,nTrain)));
  si_ = si(b_idx,:);
  
  if size(C.yTr,1)==1
    
    % Handle the case that training data is two-class with +1/-1 labels (z.Tr
    % is a row vector)
    % There is a mean Matlab bug here: For a row vector a:
    % column vector b: a(b) is a *row vector*
    % matrix        b: a(b) is a matrix of the size of b
    % This happens when calling the routine with one single test point:
    if N==1,
      out = mean(C.yTr(si_(1:C.K)),2);
    else
      % Normal case, indexing with a matrix si: everything fine here...
      out= mean(C.yTr(si_(1:C.K,:)),1);
    end
  else
    out= zeros(size(C.yTr,1),size(y,2));
    for i=1:size(C.yTr,1)
      for j=1:size(y,2)
	out(i,j) = mean(C.yTr(i,si_(1:C.K,j)), 2);
      end
    end
  end
end % End Bagging

