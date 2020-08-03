function plots(x, varargin)

if nargin>1 & ~ischar(varargin{1}),
  y= varargin{1};
  plot(squeeze(x), squeeze(y), varargin{2:end});  
else
  plot(squeeze(x), varargin{:});
end
