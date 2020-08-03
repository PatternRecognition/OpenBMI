function C= train_MSM1(xTr, yTr)
%C= train_MSM1(xTr, yTr)
%
% linear programming discrimination, multisurface method 1-norm,
% cf. Bennett, Mangasarian, Robust linear programming ...

start_cplex;

if size(yTr,1)==2,
  yTr= [-1 1]*yTr;
end

[dim, nEvents]= size(xTr);

INF= inf;
%    ga      w                     u                    v   
LB= [-INF;  -INF*ones(dim,1);  zeros(nEvents,1);     zeros(nEvents,1);   ];
UB= [INF;   INF*ones(dim,1);   INF*ones(nEvents,1);  INF*ones(nEvents,1);];
A= sparse([yTr', -diag(yTr)*xTr', -diag(yTr==1),-diag(yTr==-1)]);

bb= -ones(nEvents,1);

m= sum(yTr==1);
k= sum(yTr==-1);
f = [0; zeros(dim,1); ones(nEvents,1)/m;ones(nEvents,1)/k];
res= lp_solve(LPENV, f, A, bb, LB, UB, 0, 0,'dual') ;

C.b= -res(1);
C.w= res(1+(1:dim));
%C.u= res(1+dim+(1:nEvents));
%C.v= res(1+dim+nEvents+(1:nEvents));
