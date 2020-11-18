function datadim = util_getDataDimension(dat)
%GETDATADIMENSION - determines whether (disregarding channels) the 
% dataset in DAT is 1D or 2D, using heuristics. 1D data refers to time 
% or frequency data, 2D to time x frequency data.
%
%Usage:
% dim = util_getDataDimension(DAT)
%
%Input:
% DAT  - Struct of data
%
%Output:
% datadim - dimensionality (1 or 2) of the data

% Matthias Treder

ndim = ndims(dat.x);

if ndim==2
  datadim = 1;
  
elseif ndim==3 
  % Could be 1D or 2D (time-frequency) data
  nchan = numel(dat.clab);
  ss = size(dat.x);
  if ss(2)==nchan && ss(3)==nchan
      % This is the ambiguous case because it could be epoched 1D data with
      % second dimension=channels and 3rd dimension=epochs
      % or 
      % non-epoched 2D data with second dimension frequencies and 3rd
      % dimension=channels.
      % Try to solve using heuristics
      if isfield(dat,'y') && size(dat.y,1)==ss(3)  
        datadim = 1;
      else
        % assume 2D
        datadim = 2;
      end
  elseif ss(2)==nchan
    % 3rd dimension should be epochs
    datadim = 1;
  elseif ss(3)==nchan
    datadim = 2;
  else
    error('Number of channels in clab field (%d) does not match size of second [%d] or third dimension [%d] of .x.\n', ...
      nchan,size(dat.x,2),size(dat.x,3))
  end
  
elseif ndim==4
  datadim = 2;
  
else
  error('%d data dimensions not supported.\n',ndim)
end