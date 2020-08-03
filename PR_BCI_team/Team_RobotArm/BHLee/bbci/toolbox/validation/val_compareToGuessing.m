function [signif, range] = val_compareToGuessing(label, out, varargin);
%val_compareToGuessing compares output of some classification to 
%randomly chosen label in sense of significance.
%
% usage:
%   [signif, range] = val_compareToGuessing(label,out,<opt>);
%
% input:
%   label     - true class labels, can also be a data structure (like epo)
%               including label field '.y'
%   out       - classifier output (as given, e.g., by the third output
%                argument of xvalidation). Is only used as sign(out) 
%                or max(out)
%   opt       - a struct with fields (or directly the alpha value)
%           .alpha:   an array of significance levels <default: 0.05>
%           .priors:  an array of class priors, should sum to one, or it 
%                     can be 0 to use the priors given by the label 
%                     structure. <default: 0>
%
% output:
%   signif:   a column vector with length length(opt.alpha), which gives 
%             for each value of opt.alpha a 
%                 1: if classifier is significantly better ...
%                -1: if classifier is significantly worse ...
%                 0: if classifier is significantly neither better nor worse...
%             than chance of level alpha.
%   range:    a length(opt.alpha)x2 matrix which gives information about 
%             the ranges where classification is significantly better, 
%             worse or "equal" than chance.
%
% Note:
% This function is typically used for outputs from a leave-one-out
% validation (xvalidation with opt.sample_fcn= 'leaveOneOut').
% For outputs from a xvalidation (with >1 partitionings)
% you get significance values for each partitioning, which you have to
% interpret yourself if they are not consistent.
%
% Guido Dornhege, 10/03/2004

if length(varargin)==1 & ~isstruct(varargin{1}),
  opt= struct('alpha', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end

opt = set_defaults(opt, 'alpha',0.05, 'priors',0);

%% convert label format to vector of label indices
if size(label,1)==1
  label = (label>0)+1;
else
  [dum,label] = max(label,[],1);
end

%% convert continuous classifier output to (crisp) estimated labels
out= out2label(out);
  
if opt.priors==0
  opt.priors = hist(label,min(label):max(label));
end
opt.priors = opt.priors/sum(opt.priors);


range = zeros(length(opt.alpha),2);
n = length(label);
p = 0;

be = [0;0];

while min(be(:,end))<=max(opt.alpha)
  v = betainc([1-opt.priors;opt.priors],n-p,p+1);
  p = p+1;
  be(:,p) = v*opt.priors';
end

if length(opt.alpha)==1
  ran = (be<=opt.alpha);
  ran1 = max(find(ran(1,:)));
  ran2 = max(find(ran(2,:)));
  range = [ran1-1,n+1-ran2];
else
  ran = repmat(permute(be,[3 2 1]),[length(opt.alpha),1,1])<=repmat(opt.alpha',[1,size(be,2),2]);
  ran = diff(ran,1,2);
  [dum,ran] = min(ran,[],2);
  range = ran(:,:);
  range(:,1) = range(:,1)-1;
  range(:,2) = n+1-range(:,2);
end

range = range/n;

if size(out,1)==1,
  mis = mean(out~=label, 2);
  signif = -1+(mis<=range(:,2))+(mis<=range(:,1));
else
  msg= 'See ''Note'' in the help of this function.';
  bbci_warning(msg, 'bbci:validation', mfilename);
  mis = mean(out~=repmat(label,[size(out,1) 1]), 2);
  range= range';
  Mis = repmat(mis, [1 size(range,2)]);
  oo = ones(size(out,1), 1);
  signif = -1+(Mis<=oo*range(2,:))+(Mis<=oo*range(1,:));
end
