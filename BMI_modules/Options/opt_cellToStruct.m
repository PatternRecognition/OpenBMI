function [ opt ] = opt_cellToStruct( varargin )
%OPT_CELLTOSTRUCT Summary of this function goes here
%   Detailed explanation goes here
if nargin==0,
  return;
end

if isstruct(varargin)
    opt=varargin;
elseif iscell(varargin) % cell to struct
    [nParam temp]=size(varargin{1});
    for i= 1:nParam
        str = varargin{1}{i,1};
        if ~ischar(str),
            error('Invalid parameters: str must be string');
        end
        opt.(str)= varargin{1}{i,2};
    end
    if nParam == 0
        opt = struct({});
    end
end

end

