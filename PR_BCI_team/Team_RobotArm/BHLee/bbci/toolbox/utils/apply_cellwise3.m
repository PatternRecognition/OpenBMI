function out= apply_cellwise3(varargin)

out_cell= apply_cellwise(varargin{:});
out= reshape([out_cell{:}], size(varargin{1}));
