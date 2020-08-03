function out= apply_cellwise(C, fcn, varargin)
%out= apply_cellwise(C, fcn, params)
%
% applies the function 'fcn' to each cell of the cell array C.

out= cell(size(C));
for ii= 1:length(C(:)),
  out(ii)= {feval(fcn, C{ii}, varargin{:})};
end
