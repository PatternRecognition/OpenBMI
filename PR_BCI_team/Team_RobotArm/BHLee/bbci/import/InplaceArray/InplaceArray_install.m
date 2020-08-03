function InplaceArray_install
% function InplaceArray_install
% Installation by building the C-mex files for InplaceArray package
%
% Author Bruno Luong <brunoluong@yahoo.com>
% Last update: 28-Jun-2009 built inplace functions

arch=computer('arch');
mexopts = {'-v' '-O' ['-' arch]};
% 64-bit platform
if ~isempty(strfind(computer(),'64'))
    mexopts(end+1) = {'-largeArrayDims'};
end

% Internal representation of mxArray
buildInternal_mxArrayDef('Internal_mxArray.h');

% Inplace tool
mex(mexopts{:},'inplacearray.c');
mex(mexopts{:},'releaseinplace.c');