
function dat = epoToCnt(dat)
%
%  function cnt = epoToCnt(epo)
%
%  Concatenates the epochs chronologically, i.e. reshapes the data (field
%  .x) from 3D to 2D. The class labels (field .y) are removed.
%
%  IN:     dat     -     epoched data (3D)
%  OUT:    dat     -     data in continous format (2D)
%
%  Simon Scholler, 15.5.2009
  
  if length(size(dat.x))~=3
    error('Epoched data structure expected.')
  end
  
  [T nCh nEpo] = size(dat.x);
  dat.x = reshape(permute(dat.x,[1 3 2]), [T*nEpo nCh]);
  if isfield(dat,'y')
      rmfield(dat,'y');
  end