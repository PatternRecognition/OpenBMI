function [C, J] = train_FisherDiscriminant2(xTr, yTr, scaled)
% [C, J] = train_FisherDiscriminan2(xTr, yTr);
% Trains the usual Fisher Discriminant for two classes
% Second output argument J is the value of Fisher objective function, 
%   J = (m1-m2)^2/(v1+v2),
% where m1 and m2 are the projected mean values of the data, and v1, v2
% are the variances of the projected data

if nargin<3,
  scaled= 0;
end

% Make things work even when labels are given in +1/-1 format
if size(yTr, 1) == 1,
  yTr= [yTr<0; yTr>=0];
end

c1= find(yTr(1,:));
c2= find(yTr(2,:));

C.mean= [mean(xTr(:,c1),2) mean(xTr(:,c2),2)];
%Sb= (m1-m2)*(m1-m2)';     %% between-class scatter

p1= length(c1);
p2= length(c2);
Xnorm1= xTr(:,c1) - C.mean(:,1)*ones(1,p1);
Xnorm2= xTr(:,c2) - C.mean(:,2)*ones(1,p2);
Xnorm= [Xnorm1, Xnorm2];

if size(Xnorm,1)>size(Xnorm,2),
  [U,S,V]= svd(Xnorm);
else
  [U,S,V]= svd(Xnorm*Xnorm');
end
if size(Xnorm,1)==1,
  diagS= S(1,1);
else
  diagS= diag(S);
end
tol= max(size(Xnorm)) * eps*max(diagS);
r= sum(diagS>tol);
if r==1,
  %% Why do we need this special case?
  Xred= U(:,1) * (1/S(1,1)) * U(:,1)';
else
  Xred= U(:,1:r) * diag(1./diag(S(1:r,:))) * U(:,1:r)';
end 

S1= Xnorm1*Xnorm1';
S2= Xnorm2*Xnorm2';
%Sw= S1 + S2;               %% within-class scatter

C.w= Xred*diff(C.mean, [], 2);
if scaled,
  %% Scale classifier weight such that the distance between the projected
  %% class means is 2. If 'boundary light' (see below) is chosen, then
  %% the class means are projected to -1 and 1.
  C.w= C.w/(C.w'*diff(C.mean, [], 2))*2;
end
%C.b= -C.w'*sum(C.mean,2)/2; %% boundary light (assumes equal var of projections)

%% Estimate bias as optimal cut between the projected class distributions:
u1= C.w'*C.mean(:,1);        %% estimate mean of projections
u2= C.w'*C.mean(:,2);
s1q= C.w'*S1*C.w/(p1-1);   %% estimate variance of projections
s2q= C.w'*S2*C.w/(p2-1);

J= ((u2-u1).^2)./(s1q+s2q);

prior1= 0.5;               %% equal prior distribution
prior2= 0.5;               %% you can also use p1, p2 as priors (i.e., priors
                           %% estimated from the training set)
a= s2q - s1q;
b= 2*( u2*s1q - u1*s2q );
c= u1^2*s2q - u2^2*s1q - s1q*s2q*(2*log(prior1/prior2)-log(s1q/s2q));

C.b= 2*c / ( b + sqrt(b^2-4*a*c) );
%b2 = (b+sqrt(b^2-4*a*c))/(2*a);  
