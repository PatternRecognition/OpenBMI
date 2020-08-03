function C= train_kNearestNeighbor(xTr, yTr, K)
%C= trainNearest(xTr, yTr, K)
% train a classifier, where a data point is assigned to a class 
% most of his K nearest neighbors are.
% 
% input: K   the number of used neighbors ( a positive number)

C.xTr= xTr;
if size(yTr,1) ==2
    C.yTr = [-1 1]*yTr;
else
    C.yTr = yTr;
end
C.K= K;
