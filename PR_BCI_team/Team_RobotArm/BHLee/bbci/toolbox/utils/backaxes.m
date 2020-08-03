function h_out= backaxes(varargin)

vis= get(gcf, 'Visible');
if nargin==1,
  axes(varargin{:});
else
  out= axes(varargin{:});
end
set(gcf, 'Visible',vis);

if nargout>0,
  h_out= out;
end
