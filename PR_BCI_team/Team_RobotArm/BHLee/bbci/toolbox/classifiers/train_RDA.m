function C= train_RDA(xTr, yTr, la, ga, priorP)
%C= train_RDA(xTr, xTr, <lambda=0, gamma=0, priorP>)
%
% IN  xTr    - training samples
%     xTr    - label of the training samples, [nClasses nSamples], or
%              with range {-1,1} for 2-class problems
%     lambda - value in [0 1], morphing between QDA (0) and LDA (1).
%     gamma  - regularization: shrinkage of covariance matrices,
%            - 0 means unregularized, 1 assumes spherical covariance matrices.
%            - 'auto' starts implicit regularization by shrinkage
%     priorP - class priors, default is equal prior for all classes
%              (i.e. 1/nClasses), using '*' selects the class priors as 
%              found in the training set.
%
% OUT C      - trained classifier (apply with apply_separatingHyperplane)
%
% SEE apply_separatingHyperplane
%
% The class covariance matrix Sigma_i for class i is modified by 
%    Sigma_i= (1-la)*Sigma_i + la*Sqa;
%    Sigma_i= (1-ga)*Sigma_i + ga/d*trace(Sigma_i)*eye(d);
% where d is the dimensionality of the space, and Sqa the overall covariance
%
% REF J.H. Friedman, "Regularized Discriminant Analysis", 
%                    J Am Stat Assoc 84(405), 1989.


if ~exist('la','var') | isempty(la)
  la = 0;
end
if ~exist('ga','var') | isempty(ga)
  ga = 0;
end


if size(yTr,1)==1,
  nClasses= 2;
  clInd{1}= find(yTr==-1);
  clInd{2}= find(yTr==1);
  N= [length(clInd{1}) length(clInd{2})];
else
  nClasses= size(yTr,1);
  clInd= cell(nClasses,1);
  N= zeros(nClasses, 1);
  for ci= 1:nClasses,
    clInd{ci}= find(yTr(ci,:));
    N(ci)= length(clInd{ci});
  end
end

if ~exist('priorP', 'var'),
  priorP = ones(nClasses,1)/nClasses;
elseif isequal(priorP, '*'),
  priorP = N/sum(N);
end
 
d= size(xTr,1);
C.w= zeros(d, nClasses);
C.b= zeros(1, nClasses);
C.sq= zeros(d, d, nClasses);
Sqa= zeros(d, d);
for ci= 1:nClasses,
  cli= clInd{ci};
  C.w(:,ci)= mean(xTr(:,cli), 2);
  yc= xTr(:,cli) - C.w(:,ci)*ones(1,N(ci));
  if strcmp(ga,'auto'),   %% Added by Michael and Benjamin: use shrinkage
      Sq=clsutil_shrinkage(yc);
  else
      Sq= yc*yc';
  end
  Sqa= Sqa + Sq;
  C.sq(:,:,ci)= Sq / (N(ci)-1);
end
Sqa= Sqa / (sum(N)-1);
C.b= zeros(1, nClasses);
for ci= 1:nClasses,
  if la~=0,
    S= (1-la)*C.sq(:,:,ci) + la*Sqa;
  else
    S= C.sq(:,:,ci);
  end
  if ~strcmp(ga,'auto'),  %% Only use regularization parameter if automatic shrinkage is off
      if ga~=0,
          S= (1-ga)*S + ga/d*trace(S)*eye(d);
      end
  end
  S= pinv(S);
  C.sq(:,:,ci)= -0.5*S;
  C.b(ci)=  -0.5*C.w(:,ci)' * S*C.w(:,ci) + ...
            0.5*log(max([det(S),realmin])) + log(priorP(ci));
  C.w(:,ci)= S*C.w(:,ci);
end
C.b=C.b';

if nClasses==2,
  sq(:,:)= C.sq(:,:,2) - C.sq(:,:,1);
  C.sq= sq;
  C.w= C.w(:,2)-C.w(:,1);
  C.b= C.b(2) - C.b(1);
end
