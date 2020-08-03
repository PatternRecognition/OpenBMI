function signif = val_compareClassifiers(mis1,mis2,varargin);
%VAL_COMPARECLASSIFIER compares two classifiers
%
% usage: 
%   signif = val_compareClassifiers(mis1,mis2,<opt>);
%
% input:
%   mis1  :  error rate for independent sets of classifier 1
%   mis2  :  error rate for the same independent sets of classifier 2
%   opt   :  a struct (or only field alpha):
%         .alpha:  an array of signifance levels (default [0.05])
%         .type:   'wilcoxon' or 'chance' (default is the latter)
%
% output:
%   signif:  an array regarding opt.alpha with values:
%             1  : if class1 better than class2
%             -1 : if class1 worse than class2
%             0  : otherwise
%
% Guido Dornhege, 10/03/2004

if length(varargin)==1 & ~isstruct(varargin{1}),
  opt= struct('alpha', varargin{1}),
else
  opt= propertylist2struct(varargin{:});
end

opt = set_defaults(opt,'alpha',0.05, 'type','chance');

switch lower(opt.type);
 case 'chance'
  range = zeros(length(opt.alpha),1);
  n = length(mis1);
  p = 0;
  
  be = [-1];
  
  while be(end)<=max(opt.alpha)
    be(p+1) = betainc(0.5,n-p,p+1);
    p = p+1;
  end
  
  if length(opt.alpha)==1
    p = p-1;
  else
    p = repmat(be,[length(opt.alpha),1])<=repmat(opt.alpha',[1,length(be)]);
    p = diff(p,1,2);
    [dum,p] = min(p,[],2);
  end
  signif = zeros(length(opt.alpha),1);
  signif(find(sum(mis1<mis2)<=p))=1;
  signif(find(sum(mis1>mis2)<=p))=-1;
  
 case 'wilcoxon'
  n = length(mis1);
  m = length(mis2);
  [dum,ind] = sort([mis1,mis2]);
  indn = ind(1:n);
  indm = ind(n+1:n+m);
  if n<=10 & m<=10
    comb = nchoosek(1:n+m,n);
    comb = sum(comb,2);
    vert = hist(comb,1:max(comb));
    vert = vert/sum(vert);
    vert = cumsum(vert);
    if length(opt.alpha)==1
      [dum,ver] = max(find(vert<=opt.alpha));
      [dum,vert] = min(find(1-vert<=opt.alpha));
      range = [ver,vert];
    else
      range = zeros(length(opt.alpha),2);
      dif = repmat(vert,[length(opt.alpha),1])<=repmat(opt.alpha',[1,length(vert)]);
      [dum,range(:,1)] = min(diff(dif),[],2);
      dif = repmat(1-vert,[length(opt.alpha),1])<=repmat(opt.alpha',[1,length(vert)]);
      [dum,range(:,2)] = min(diff(dif),[],2);
    end    
  else
    n = length(mis1);
    m = length(mis2);
    ew = n*(m+n+1)*0.5;
    va = m*n*(m+n+1)/12;
    range = zeros(length(opt.alpha),2);
    range(:,1) = ew+sqrt(2).*sqrt(va).*erfinv(1-2*opt.alpha');
    ew = m*(m+n+1)*0.5;
    range(:,2) = ew+sqrt(2).*sqrt(va).*erfinv(1-2*opt.alpha');
  end
  signif = (sum(indn)<=range(:,1))-(sum(indm)<=range(:,2));
end

