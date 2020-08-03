function uniyLim= unifyYLim(h, varargin)
%UNIFYYLIM - Selects a common YLim for a set of axes
%
%Description:
% unifyYLim changes the YLim setting in axes handled by "hAxes", 
% or (default) in all children of the current figure.
%
%Synopsis:
% uniYLim= unifyYLim(<H, OPT>)
%
%Input:
% H   - vector of axes handles, default: children of current figure.
% OPT - struct or property/value list of optional properties:
%  .policy - specifies how to choose the YLim: 'auto' uses the
%            usual mechanism of Matlab; 'tightest' (default) selects
%            YLim as the exact data range; 'tight' takes the data range,
%            adds a little border and selects 'nice' limit values.
%  .tighten_border: for OPT.policy='tight' a border is added to the
%            data range. The size of the border is the fraction
%            OPT.tighten_border of the whole range, default 0.03.
%  .symmetrize: choose limits that are symmetric around 0.
%
%Output:
% uniYLim - selected YLim

% Author(s): Benjamin Blankertz Aug 2000

if ~exist('h','var') | isempty(h),
  h= get(gcf, 'children');
end

if length(varargin)==1,
  opt= struct('policy', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'policy', 'auto');
%% for compatibility
if isequal(opt.policy, 1),
  opt.policy= 'tight';
end

for hi= 1:length(h),
  isaxes(hi)= strcmp(get(h(hi), 'type'), 'axes') & ...
      ~strcmp(get(h(hi), 'tag'), 'legend');
end
h= h(find(isaxes));

for hi= 1:length(h),
  yLim(hi,:)= selectYLim(h(hi), opt, 'setLim',0);
end
uniyLim= [min(yLim(:,1)) max(yLim(:,2))];
set(h, 'yLim',uniyLim); 

if nargout==0,
  clear uniyLim;
end
