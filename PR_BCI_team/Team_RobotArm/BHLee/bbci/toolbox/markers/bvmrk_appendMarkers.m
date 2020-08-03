function mrk= bvmrk_appendMarkers(varargin)

if nargin==1 & iscell(varargin{1}),
  mrk= cat(1, varargin{1}{:});
end

mrk= cat(1, varargin{:});
