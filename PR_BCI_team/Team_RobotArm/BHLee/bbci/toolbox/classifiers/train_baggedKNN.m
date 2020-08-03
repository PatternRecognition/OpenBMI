function C = train_baggedKNN(xTr, yTr, K, nBagg)
% C = train_baggedKNN(xTr, yTr, K, nBagg)
%
% Train a classifier, where a data point is assigned to a class 
% where most of its K nearest neighbors are - repeated nBagg times over
% bootstraps of the training set.
% 
% input: K     the number of used neighbors ( a positive number).
%        nBagg number of bagging iterations (default 50).
%
% Author: Sebastian Mika, idalab GmbH, (c) 2005
% $Revision: 1.2 $
  

C.xTr= xTr;
if size(yTr,1) ==2
    C.yTr = [-1 1]*yTr;
else
    C.yTr = yTr;
end
C.K= K;

if nargin < 4
  C.B = 50;
else
  C.B = nBagg;
end
