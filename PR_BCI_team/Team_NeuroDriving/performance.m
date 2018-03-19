function [X,Y,T,AUC] = performance(pred, testLabel)

pTarget = find(pred > 0);
pNon = find(pred <= 0);
pred(pTarget) = 1; pred(pNon) = 0;
LabelVector = zeros(1,length(testLabel));
LabelVector(find(testLabel(2,:)==1))=1;

% AUC
[X,Y,T,AUC]=perfcurve(LabelVector,pred,1);

% % ACC
% correct = find(pred==LabelVector);
% p= length(correct) / length(pred) * 100;

