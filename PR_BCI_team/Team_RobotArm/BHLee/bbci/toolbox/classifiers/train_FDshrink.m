function [C, J] = train_FDshrink(xTr, yTr, varargin)
%TRAIN_FDSHRINK - Fisher's Discriminant with Covariance Shrinkage
%
%Synopsis:
% [C, J] = train_FDshrink(X, Y);
%
%Arguments:
% X: Data samples [nDim nSamples], each column is one sample
% Y: Labels
%
%Returns:
% C: trained classifier struct (format of separating hyperplane)
% J: value of Fisher objective function, 
%      J = (m1-m2)^2/(v1+v2),
%    where m1 and m2 are the projected mean values of the data, and v1, v2
%    are the variances of the projected data
%
%See also: clsutil_shrinkage,
%   train_FisherDiscriminant (multi-class),
%   apply_separatingHyperplane, 
%   trainClassifier, applyClassifier

% Make things work even when labels are given in +1/-1 format
if size(yTr, 1) == 1,
  yTr= [yTr<0; yTr>=0];
end

c1= find(yTr(1,:));
c2= find(yTr(2,:));

m1= mean(xTr(:,c1),2);
m2= mean(xTr(:,c2),2);
%Sb= (m1-m2)*(m1-m2)';     %% between-class scatter

p1= length(c1);
p2= length(c2);
Xnorm1= xTr(:,c1) - m1*ones(1,p1);
Xnorm2= xTr(:,c2) - m2*ones(1,p2);

Sw= clsutil_shrinkage([Xnorm1, Xnorm2], varargin{:});

C.w= pinv(Sw)*(m2-m1);
%C.b= -C.w'*(m1+m2)/2;     %% boundary light (assumes equal var of projections)

%% Estimate bias as optimal cut between the projected class distributions:
u1= C.w'*m1;               %% estimate mean of projections
u2= C.w'*m2;
S1= Xnorm1*Xnorm1';
S2= Xnorm2*Xnorm2';
s1q= C.w'*S1*C.w/(p1-1);   %% estimate variance of projections
s2q= C.w'*S2*C.w/(p2-1);

J = ((u1-u2).^2)./(s1q+s2q);

prior1= 0.5;               %% equal prior distribution
prior2= 0.5;               %% you can also use p1, p2 as priors (i.e., priors
                           %% estimated from the training set)
a= s2q - s1q;
b= 2*( u2*s1q - u1*s2q );
c= u1^2*s2q - u2^2*s1q - s1q*s2q*(2*log(prior1/prior2)-log(s1q/s2q));

C.b= 2*c / ( b + sqrt(b^2-4*a*c) );
%b2 = (b+sqrt(b^2-4*a*c))/(2*a);  
