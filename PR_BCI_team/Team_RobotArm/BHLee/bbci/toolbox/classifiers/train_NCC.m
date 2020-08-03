function C = train_NCC(xTr, yTr, gamma)
% TRAIN_NCC - Train Nearest Centroid Classifier (NCC)
%
% Usage:
%   C = TRAIN_RLDA(X, LABELS)
%
% Input:
%   X: Data matrix, with one point/example per column. 
%   LABELS: Class membership. LABELS(i,j)==1 if example j belongs to
%           class i.
% Output:
%   C: Classifier structure, hyperplane given by fields C.w and C.b


mu1= mean(xTr(:,find(yTr(1,:))),2);
mu2= mean(xTr(:,find(yTr(2,:))),2);

C.w = mu2 - mu1;
C.b = -0.5*C.w'*(mu1+mu2);
