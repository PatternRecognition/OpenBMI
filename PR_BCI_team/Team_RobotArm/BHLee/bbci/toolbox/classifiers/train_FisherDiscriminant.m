function C= train_FisherDiscriminant(xTr, yTr)
%C = train_FisherDiscriminant(xTr,yTr);
% Trains the usual Fisher Discriminant (multi-class).
% see train_FisherDiscriminant2 for two class

nClasses= size(yTr,1);

if nClasses>2,
  clInd= cell(nClasses,1);
  N= zeros(nClasses, 1);
  for ci= 1:nClasses,
    clInd{ci}= find(yTr(ci,:));
    N(ci)= length(clInd{ci});
  end
  
  priorP = ones(1,nClasses)/nClasses; 
  M= mean(xTr,2);
  d= size(xTr,1);
  Sb= zeros(d, d); 
  S= zeros(d, d, nClasses); 
  for ci= 1:nClasses,
    cli= clInd{ci};
    m(:,ci)= mean(xTr(:,cli),2);
    yc= xTr(:,cli) - m(:,ci)*ones(1,N(ci));
    S(:,:,ci)= yc*yc';
    Sb= Sb + N(ci)*(m(:,ci)-M)*(m(:,ci)-M)';
  end
  
  Sw= sum(S, 3);
  
  [V,D]= eig(Sb, Sw);
  [dummy, si]= sort(-diag(D));
  w= V(:,si(1:nClasses-1));
  proSW = pinv(w'*Sw*w);
  ch = w*proSW*w';
  C.w = ch*m;
  for i= 1:nClasses,
    cl1= clInd{i};
    cl2= find(~yTr(i,:));
    u1= mean(C.w(:,i)'*xTr(:,cl1));
    u2= mean(C.w(:,i)'*xTr(:,cl2));
    s1q= var(C.w(:,i)'*xTr(:,cl1));
    s2q= var(C.w(:,i)'*xTr(:,cl2));
    
    a= s2q - s1q;
    b= 2*( u2*s1q - u1*s2q );
    c= u1^2*s2q - u2^2*s1q + s1q*s2q*log(s1q/s2q);
    
    if u1<u2,
      C.b(i,1) =2*c / ( b + sqrt(b^2-4*a*c) );
    else
      C.b(i,1) = (b+sqrt(b^2-4*a*c))/(2*a);  
    end
  end
  
else
  
  C = train_FisherDiscriminant2(xTr,yTr);
end
