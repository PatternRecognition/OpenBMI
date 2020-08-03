function out= ml_intersect(varargin)

if nargin==2,
  a= varargin{1};
  out= a(find(ismember(a, varargin{2})));
elseif nargin==1,
  out= varargin{1};
else
  out= ml_intersect(ml_intersect(varargin{1}, varargin{2}), varargin{3:end});
end
