function datadim = getDataDimension(dat)
%GETDATADIMENSION - determines whether (disregarding channels) the 
% dataset in DAT is 2D or 3D, using heuristics. 2D data refers to time x
% amplitude or frequency x amplitude data, 3D to frequency x time x 
% amplitude data.
%
%Usage:
% dim = getDataDimension(DAT)
%
%Input:
% DAT  - Struct of data
%
%Output:
% datadim - dimensionality (2 or 3) of the data

ndim = ndims(dat.x);

if ndim==2
  datadim = 2;
  
elseif ndim==3 
  % Could be 2D or 3D (time-frequency) data
  nt = numel(dat.t);
  nc = numel(dat.clab);
  ss = size(dat.x);
  if ss(1)==nt && ss(2)==nc
    datadim = 2;
  elseif ss(2)==nt && ss(3)==nc
    datadim = 3;
  else
    error('Number of elements in t (%d) and clab (%d) does not match with size of x (%s).\n', ...
      nt,nc,num2str(size(dat.x)))
  end
  
elseif ndim==4
  datadim = 3;
  
else
  error('%d data dimensions not supported.\n',ndim)
end

