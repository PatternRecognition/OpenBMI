function out= apply_on_cell(C, fcn, varargin)
%out= apply_on_cell(C, fcn, params)
%
% applies the function 'fcn' to each cell of the cell array C.
% fcn must return a number

out= zeros(size(C));
for ii= 1:length(C(:)),
  out(ii)= feval(fcn, C{ii}, varargin{:});
end
