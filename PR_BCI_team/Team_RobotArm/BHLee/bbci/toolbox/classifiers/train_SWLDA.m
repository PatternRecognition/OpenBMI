% TRAIN_SWLDA - Train stepwise linear discriminant analysis (SWLDA)
%
% Usage:
%   C = TRAIN_SWLDA(X, LABELS, MAXVAR)
%   C = TRAIN_SWLDA(X, LABELS, OPTS)
%
% Input:
%   X: Data matrix, with one point/example per column. 
%   LABELS: Class membership. LABELS(i,j)==1 if example j belongs to
%           class i.
%   MAXVAR: maximum number of selected variables by stepwise LDA.
%   PENTRY: p-value to enter regression (default: 0.1)
%   PREMOVAL: variables are eliminated if p-value of partial f-test 
%             exceeds pRemoval (default: 0.15)
%   OPTS: options structure as output by PROPERTYLIST2STRUCT. Recognized
%         options are:
%         'maxVar':  
%         'pEntry': 
%         'pRemoval':  see above.
% Output:
%   C: Classifier structure, hyperplane given by fields C.w and C.b
%
% Description:
%   TRAIN_SWLDA trains a stepwise LDA classifier on data X with class
%   labels given in LABELS. The stepwise procedure stops when there 
%   are no more variables falling below the critical p-value of PENTRY.
%   It can be limited by MAXVAR, the maximum number of variables
%   that should be selected. The critical p-value for removal is
%   set to PREMOVAL.
%
%
%   References: N.R. Draper, H. Smith, Applied Regression Analysis, 
%   2nd Edition, John Wiley and Sons, 1981. This function implements
%   the algorithm given in chapter 'Computional Method for Stepwise 
%   Regression' of the first edition (1966).
%
% Example:
%   train_SWLDA(X, labels, 12, 0.1, 0.15)
%   train_SWLDA(X, labels, ...
%               propertylist2struct('maxVar',12,'pEntry',0.1,'pRemoval',0.15))
%   
%   
%   See also APPLY_SEPARATINGHYPERPLANE,TRAIN_LDA,
%
function [C,maxVar] = train_SWLDA(xTr, yTr, maxVar, pEntry, pRemoval)

% Standard input argument checking
error(nargchk(2, 5, nargin));

% No, though shallst not use 'exist' for variables
if nargin<3 | isempty(maxVar),
  maxVar = size(yTr,2);
end
if nargin<4 | isempty(pEntry),
  pEntry = 0.1;
end
if nargin<5 | isempty(pRemoval),
  pRemoval = 0.15;
end
% Now comes the new part, where the options can also be passed as an options
% structure. Check whether MAXVAR has been created by PROPERTYLIST2STRUCT:
if ispropertystruct(maxVar),
  if nargin>5,
    error('With given OPTS, no additional input parameter is allowed');
  end
  % OK, so the third arg was not maxVar, but the options structure
  opt = maxVar;
  % Set default parameters
  opt = set_defaults(opt, 'maxVar', size(yTr,2));
  % Extract parameters from options
  maxVar = opt.maxVar;
  pEntry = opt.pEntry;
  pRemoval = opt.pRemoval;
end

if maxVar<1 | maxVar>size(yTr,2),
  error(['limiting parameter of setpwise procedure MAXVAR must be between 1 and ' num2str(size(yTr,2))]);
end

if size(yTr,1) == 1 yTr = [yTr<0; yTr>0]; end
ind = find(sum(abs(xTr),1)==inf);
xTr(:,ind) = [];
yTr(:,ind) = [];

x=xTr;

xr = x';
y = yTr(1,:)';
subset = [];

dat = [xr y];

n = size(dat,1);   % number of trials
k = size(dat,2);   % number of independent variables + 1 (response)
C.w = zeros(k-1,1);
SSu = sum(dat,1);  % uncorrected sums of squares
SCu = dat'*dat;    % uncorrected cross-products

% corrected cross-products
SCc = SCu - SSu'*SSu./n;
s2 = diag(SCc);

% correlation coefficients
R = SCc ./ sqrt(s2*s2');

A = [R [eye(k-1); zeros(1,k-1)]; [-1.*eye(k-1) zeros(k-1,1) zeros(k-1,k-1)]];

dorun=true; 
while((size(subset,2) < maxVar) & dorun)

  % statistics
  for i=1:k-1
    V(i) = A(i,k)*A(k,i)/A(i,i);
  end

  % best variable to enter regression
  fi = find(V==max(V));
  dfres = n - 2 - size(subset,2);

  % MSreg = A(fi,k)^2;
  % MSres = (A(k,k)^2 - A(fi,k)^2) / dfres;

  F = dfres*V(fi) / (A(k,k)-V(fi));
  p = ones(size(F,1),size(F,2)) - proc_fcdf(F,size(subset,2)+1,dfres);

  if(p < pEntry) 

    subset = union(subset,fi); 

    B = A -  A(:,fi)*A(fi,:)./A(fi,fi); 
    B(fi,:) = A(fi,:)/A(fi,fi);

  else
    B = A;
    dorun = false;
  end

  % test for elimination of variables already in regression
  if(size(subset,2)>1)
    old = setdiff(subset,fi);
    for i=old

      Fp = dfres*(B(i,k)^2) / (B(k,k)*B(k+i,k+i));
      pp = ones(size(Fp,1),size(Fp,2)) - proc_fcdf(Fp,1,dfres);

      if(pp > pRemoval)   % eliminate

        subset = setdiff(subset,i);

        % adapt matrices after elimination
        A = B -  B(:,k+i)*B(k+i,:)./B(k+i,k+i);
        A(i,:) = B(i,:)/B(k+i,k+i);

        B = A;

      end
    end
  end
  A = B;



end % while ~stoppingrule


% calculate coefficients for final regression equation Y^ = C.b + C.w'+x
C.b = -mean(dat(:,k),1);
for i=subset
  C.w(i) = A(i,k)*sqrt(SCc(k,k)/SCc(i,i));
  C.b = C.b + C.w(i)*mean(dat(:,i),1);
end

C.w=-C.w;
C.b=-C.b;


end 


