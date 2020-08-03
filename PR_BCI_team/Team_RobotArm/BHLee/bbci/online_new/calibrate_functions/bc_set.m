function varargout= bc_set(varargin)
%BC_SET - Short hand for bbci_calibrate_set

varargout= cell(1, nargout);
[varargout{:}]= bbci_calibrate_set(varargin{:});
