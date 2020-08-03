function dat= proc_jumpingMax(dat, nSamples, nMeans)
%dat= proc_jumpingMax(dat, nSamples, <nMaxs>)
%
% IN   dat       - data structure of continuous or epoched data
%      nSamples  - number of samples from which the max is calculated
%                  if nSamples is a matrix maxs over the rows are given 
%                  back. nMaxs is ignored then.
%      nMeans    - number of intervals from which the max is calculated
%
% OUT  dat      - updated data structure
%
% SEE proc_jumpingMeans

 
[T, nChans, nMotos]= size(dat.x);

if length(nSamples)==1,
  
  if ~exist('nMeans','var'), nMeans= floor(T/nSamples); end
    
  dat.x= permute(max(reshape(dat.x((T-nMeans*nSamples+1):T,:,:), ... 
                 [nSamples,nMeans,nChans,nMotos]),[],1),[2 3 4 1]);

  if isfield(dat, 'fs'),
    dat.fs= dat.fs/nSamples;
  end
  if isfield(dat, 't'),
    dat.t= mean(dat.t(reshape((T-nMeans*nSamples+1):T,nSamples,nMeans)));
  end

elseif size(nSamples,1)==1,
  intervals= nSamples([1:end-1; 2:end]');
  dat= proc_jumpingMeans(dat, intervals);

else
  error('dimension error: wrong input format');

end
