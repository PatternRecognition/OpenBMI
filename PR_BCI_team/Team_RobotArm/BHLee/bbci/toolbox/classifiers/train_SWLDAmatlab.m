% TRAIN_SWLDA - Train stepwise LDA using the Matlab function stepwisefit
%
% Usage:
%   C = TRAIN_SWLDAmatlab(X, LABELS, MAXVAR)
%   C = TRAIN_SWLDAmatlab(X, LABELS, MAXVAR, <PENTRY, PREMOVAL>)
%   C = TRAIN_SWLDAmatlab(X, LABELS, OPTS)
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
function [C,maxVar] = train_SWLDAmatlab(xTr, yTr, maxVar, pEntry, pRemoval)

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
  maxVar = opt.maxVar;
  pEntry = opt.pEntry;
  pRemoval = opt.pRemoval;
end

if maxVar<1 | maxVar>size(yTr,2),
  error(['limiting parameter of setpwise procedure MAXVAR must be between 1 and ' num2str(size(yTr,2))]);
end

[b, se, pval, inmodel, stats]= ...
    stepwisefit(xTr', ([-1 1]*yTr)', 'penter',pEntry, 'premove',pRemoval, ...
                'maxiter',maxVar, 'display','off');
C.w= zeros(size(b));
C.w(inmodel)= b(inmodel);
%C.b= stats.intercept;
idx1= find(yTr(1,:));
idx2= find(yTr(2,:));
C.b = -C.w'*(mean(xTr(:,idx2),2)+mean(xTr(:,idx1),2))/2;
