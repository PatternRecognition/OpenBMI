function [ opt ] = opt_cellToStruct( varargin )
% opt_cellToStruct:
% 
% Description:
%     This function makes a cell type of input argument into a struct.
%   To use this function, size of the input argument must be Nx2,
%   which first column values(string) become the field name,
%   and the second column values become the contents of the field.
% 
% Example:
%     opt = opt_cellToStruct(varargin{:});
%     opt = opt_cellToStruct({'frequency',[8,13];'time',[750 3500]});
% 
% Input:
%     varargin - cell type, size of Nx2
% Output:
%     opt      - struct type, with field name of first column of varargin
%                 and corresponding value of second column of varargin
% 
% Min-Ho, Lee
% mhlee@image.korea.ac.kr
% 
if nargin==0
  opt = struct();return;
end

if isstruct(varargin)
    opt=varargin;
elseif iscell(varargin) % cell to struct
    [n_param, n_pair]=size(varargin{1});
    if n_pair ~= 2
        error('Invalid parameters: parameters are must be paired');
    end
    for i= 1:n_param
        str = varargin{1}{i,1};
        if ~ischar(str)
            error('Invalid parameters: str must be string');
        end
        opt.(lower(str))= varargin{1}{i,2};
    end
end
end
