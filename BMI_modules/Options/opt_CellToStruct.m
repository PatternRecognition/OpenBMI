function [ opt ] = opt_CellToStruct( varargin )
%OPT_CELLTOSTRUCT Summary of this function goes here
%   Detailed explanation goes here
if nargin==0,
  return;
end

for i= 1:length(varargin)/2
  str = varargin{i};
  if ~ischar(str),
    error('Invalid parameters');
  end
  opt.(str)= varargin{i+(length(varargin)/2)};  
end

end

