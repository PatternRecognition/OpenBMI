% function B = inplacearray(A, offset, sz)
% 
% B = inplacearray(A)
%  Return the inplace-array pointed to same place A
% 
% B = inplacearray(A, OFFSET)
%  B(1) is pointed at A(1+offset) (linear indexing)
%   
% B = inplacearray(A, OFFSET, SZ)
%  Specify the dimesnion of B
%  Alternate calling syntax B = inplacearray(A, OFFSET, N1, N2, ... Nn)
% 
% INPUTS
%  A; is a (full) array
%  OFFSET: scalar, offset from the first element A(1). Note that
%          overflow/negative value is allowed. OFFSET is 0 by default
%  SZ: the dimension of the inplace output, overflow allowed (!).
% OUTPUT
%  B: nd-array of the size SZ, shared the same data than A
%       B(1) is started from A(1+offset).
%
% Class supported: all numerical, logical and char 
% 
% Important notes:
% - For advanced users only!!!! In any case use at your own risk
% - use MEX function releaseinplace(B) to release properly shared-data
%   pointer before clear/reuse B.
% - All inplace variables shared data with A must be released before
%   the original array A is cleared/reused.
% 
% Compilation:
%  >> buildInternal_mxArrayDef('Internal_mxArray.h')
%  >> mex -O -v inplacearray.c % add -largeArrayDims on 64-bit computer
%
% See also: releaseinplace, inplacecolumn
% 
% Author Bruno Luong <brunoluong@yahoo.com>
% Last update: 27/June/2009
