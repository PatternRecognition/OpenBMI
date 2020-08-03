function dat= proc_logNormalize(dat)
%dat= proc_logNormalize(dat)
%
% log-normalize each epoch (time x channels)
% 
% IN   dat   - data structure of continuous or epoched data
%
% OUT  dat   - updated data structure

% bb, ida.first.fhg.de


[T, nChans, nMotos]= size(dat.x);
 
if T*nChans==1, return; end;
 
xo= zeros(size(dat.x));
for m= 1:nMotos,
  xo(:,:,m)= -log( dat.x(:,:,m) / sum(sum(abs(dat.x(:,:,m)))) );
end

dat.x= xo;
