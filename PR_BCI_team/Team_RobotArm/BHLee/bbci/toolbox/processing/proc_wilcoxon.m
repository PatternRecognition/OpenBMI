function fv_wil = proc_wilcoxon(fv,varargin)
% fv_wil = proc_wilcoxon(fv,<alpha>)
%
% wilcoxon-test value for every feature dimension.
%
% IN     fv   - data structure of feature vectors
%        alpha- significance level for hypothesis "class1<class2".
% OUT    fv_wil- data structure of wilcoxon values (one sample only)
%            .x - normalized wilcoxon ranking values
%            .crit- critical value for significance level alpha.
% SEE    proc_r_values

% kraulem 07/05

% convert input arguments into standard form
if ndims(fv.x)>2
  siz = size(fv.x);
  fv.x = reshape(fv.x,prod(siz(1:end-1)),siz(end));
end
if size(fv.y,1)~=2,  %% now doing it pairwise:
  warning('calculating pairwise wilcoxon-values');
  combs= nchoosek(1:size(fv.y,1), 2);
  for ic= 1:length(combs),
    ep= proc_selectClasses(fv, combs(ic,:));
    if ic==1,
      fv_wil= proc_wilcoxon(ep,varargin{:});
    else
      fv_wil= proc_appendEpochs(fv_wil, proc_wilcoxon(ep,varargin{:}));
    end
  end
  return; 
end

% standard: fv has two classes.

z = [fv.x(:,find(fv.y(1,:))), fv.x(:,find(fv.y(2,:)))];
[z,id] = sort(z,2);
n1 = sum(fv.y(1,:));
n2 = sum(fv.y(2,:));
wil = zeros(size(z,1),1);
for ii = 1:size(z,1)
  % find the wilcoxon rank in every feature dimension.
  wil(ii) = sum(find(id(ii,:)<=n1));
end
wil = (wil-n1*(n1+n2+1)/2)/sqrt(n1*n2*(n1+n2+1)/12);

% construct a new fv:
fv_wil= copyStruct(fv, 'x','y','className');
fv_wil.x= wil;
fv_wil.className= {sprintf('wil( %s , %s )', fv.className{1:2})};
fv_wil.y= 1;
fv_wil.yUnit= 'wil';
if nargin>1
  fv_wil.crit = normal_invcdf(1-varargin{1});
end

return