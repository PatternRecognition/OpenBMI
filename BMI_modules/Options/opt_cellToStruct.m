function [ opt ] = opt_cellToStruct( varargin )
% opt_cellToStruct: Convert a cell into a structure.
% 
% Example:
%    opt=opt_cellToStruct(varargin{:});
% Input:
%    Size of the input cell should be nx2.
% 
if nargin==0,
  opt = struct();return;
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
end

end

