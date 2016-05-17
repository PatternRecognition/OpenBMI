function C = train_wr_multiClass_hsan(xTr, yTr, idx1, idx2, idx3, pClassy, varargin)
%C = train_wr_multiClass(xTr, yTr, model, <opt>)
%
% wrapper function for classifiers to reduce multi-class problems
% to binary.
% See Reducing Multiclass to Binary: A unifying
% Approach for Margin Classifiers by Allwein, Schapire, Singer
%
% IN  model  - classifier model
%     opt - propertylist and/or struct of options
%      .policy : 'one-vs-all', 'complete', 'all-pairs', 'sparse', 'dense'
%      .coding : 'hamming', 'lessexp', 'lesssquare', 'lesshinge', 'lesslog'

[n,m] = size(xTr);  % 44x69: hsan
nClasses = size(yTr,1);
if nClasses==1
  nClasses = 2;
  yTr = [yTr<0; yTr> 0];
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'policy', diag(2*ones(1,nClasses))-ones(nClasses), ...
                  'coding', 'hamming');

[func, params]= getFuncParam(pClassy);
trainFcn= ['train_' func];

coding= opt.coding;
M= opt.policy;

if ischar(coding)
switch coding
 case 'hamming'
  coding = inline('0.5-0.5*sign(x)','x');
 case 'lessexp'
  coding = inline('exp(-x)','x');
 case 'lesssquare'
  coding = inline('(1-x).^2','x');
 case 'lesshinge'
  coding = inline('max([zeros(size(x,1),1),1-x])','x');
 case 'lesslog'
  coding = inline('log(1+exp(-2*x))','x');
 otherwise
  error('the coding is not supported');
end
end

if ischar(M)
  switch M
   case 'one-vs-all'
    M = diag(2*ones(1,nClasses))-ones(nClasses);
   case 'complete'
    M = zeros(nClasses,2^(nClasses-1));
    M(1,1:2^(nClasses-1)) = 1;
    for i =1:(nClasses-1)
      for j=1:i
	M(i+1,((j-1)*2^(nClasses-i)+1):((j-1)*2^(nClasses-i)+2^(nClasses-i-1))) =1;
	M(i+1,((j-1)*2^(nClasses-i)+2^(nClasses-i-1)+1):j*2^(nClasses-i)) =-1;
      end
    end
    M(:,1) = [];
   case 'all-pairs'
    M = zeros(nClasses, 0.5*nClasses*(nClasses-1));
    cnt = 0;
    for i=1:nClasses-1
      for j = (i+1):nClasses
	cnt=cnt+1;
	M(i,cnt)=1;
	M(j,cnt)=-1;
      end
    end
   case 'dense'
    if nClasses<5 warning('dense is not recommend for nClasses < 5'); end
    rep=100;
    fac = 4;
    Mold = zeros(nClasses, floor(fac*log2(nClasses)));
    rho = inf;
    for i=1:rep
      M = sign(randn(nClasses,floor(fac*log2(nClasses))));
      rho1 = inf;
      for j=1:floor(fac*log2(nClasses))-1
	for k =(i+1):floor(fac*log2(nClasses))
	  d = 0.5*floor(fac*log2(nClasses))-0.5*M(:,j)'*M(:,k);
	  if d<rho1 & d>0,  rho1=d; end
	end
      end
      if rho1<rho
	Mold = M;
	rho=rho1;
      end
    end
    M=Mold;
   case 'sparse'
    if nClasses<10 warning('sparse is not recommend for nClasses < 10'); end
    rep = 1000;
    fac = 4;
    Mold = zeros(nClasses, floor(fac*log2(nClasses)));
    rho = 2*fac*log2(nClasses)+1;
    for i=1:rep
      M = round(2*rand(nClasses,floor(fac*log2(nClasses))))-1;
      rho1 = inf;
      for j=1:floor(fac*log2(nClasses))-1
	for k =(i+1):floor(fac*log2(nClasses))	
	  d = 0.5*floor(fac*log2(nClasses))-0.5*M(:,j)'*M(:,k);
	  if d<rho1 & d>0, rho1=d; end	  
	end
      end
      if rho1<rho
	Mold = M;
	rho=rho1;
      end
    end 
    M = Mold;
   otherwise
    error('M is not known')
  end
end

if exist('idx1','var')
    lengidx1 = length(idx1);
    lengidx2 = length(idx2);
    lengidx3 = length(idx3);
end

for i = 1:size(M,2)
  lab = M(:,i)'*yTr;
  c = find(~(lab==0));
  if i == 1
    dat = xTr(1:lengidx1,c);
  elseif i == 2
    dat = xTr(lengidx1+1:lengidx1+lengidx2,c);
  elseif i == 3
    dat = xTr(lengidx1+lengidx2+1:end,c);
  end
  lab = lab(c);
  C.Cl(i) = feval(trainFcn, dat, lab, params{:});
end

C.func = func;
C.M = M;
C.coding = coding;
