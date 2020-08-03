function out= cell_flaten(C, dim)
%CELL_FLATEN - Flaten all layers of a cell array
%
%Synopsis:
% CFLAT= cell_flaten(C)
% CFLAT= cell_flaten(C, DIM)
%
%Arguements:
%  C: Arbitrary cell array
%  DIM: Dimension in which CFLAT should live, default 2.
%
%Returns:
% CFLAT: Cell array, where each entry in CFLAT is a non-cell object.

% blanker@cs.tu-berlin.de

if nargin<2,
  dim= 2;
end

if ~iscell(C),
  out= {C};
  return;
end

out= {};
for ci= 1:length(C),
  out= cat(dim, out, cell_flaten(C{ci}));
end
