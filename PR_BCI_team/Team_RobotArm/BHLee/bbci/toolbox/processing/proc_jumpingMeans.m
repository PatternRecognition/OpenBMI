function dat= proc_jumpingMeans(dat, nSamples, nMeans)
%dat= proc_jumpingMeans(dat, nSamples, <nMeans>)
%dat= proc_jumpingMeans(dat, intervals)
%
% IN   dat      - data structure of continuous or epoched data
%      nSamples  - number of samples from which the mean is calculated
%                  if nSamples is a matrix means over the rows are given 
%                  back. nMeans is ignored then.
%      nMeans    - number of intervals from which the mean is calculated
%      intervals - each row contains an interval (format: [start end] in ms)
%                  in which the mean is to be calculated
%                  (requires field 't' in dat)
%
% OUT  dat      - updated data structure
%
% SEE proc_jumpingMedians

% bb, ida.first.fhg.de


[T, nChans, nMotos]= size(dat.x);

if length(nSamples)==1,
  
  if ~exist('nMeans','var'), nMeans= floor(T/nSamples); end
    
  dat.x =permute(mean(reshape(dat.x((T-nMeans*nSamples+1):T,:,:), ... 
                 [nSamples,nMeans,nChans,nMotos]),1),[2 3 4 1]);

  if isfield(dat, 'fs'),
    dat.fs= dat.fs/nSamples;
  end
  if isfield(dat, 't'),
    dat.t = mean(dat.t(reshape((T-nMeans*nSamples+1):T,nSamples,nMeans)));
  end

elseif size(nSamples,1)==1 & size(nSamples,2)~=2,
  
  intervals= nSamples([1:end-1; 2:end]');
  dat= proc_jumpingMeans(dat, intervals);

else
  
  nMeans= size(nSamples ,1);
  da = zeros(nMeans, nChans, nMotos);
  for i = 1:size(nSamples,1),
    if any(isnan(nSamples(i,:))),
      da(i,:,:) = NaN;
    else
      I = find(dat.t>=nSamples(i,1) & dat.t<=nSamples(i,2));
      da(i,:,:) = mean(dat.x(I,:,:),1);
    end
  end

  dat.x = da;
  dat= rmfield(dat, 'fs');
  dat.t = mean(nSamples, 2)';

end
