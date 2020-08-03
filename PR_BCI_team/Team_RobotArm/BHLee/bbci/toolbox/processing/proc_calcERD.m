function dat= proc_calcERD(dat, refIval, smoothy)
%epo= proc_calcERD(epo, refIval, <smoothy>)
%
% calculate the ERD/ERS relative to a specified reference interval
% this function is only the last step in calculating the ERD/ERS:
% before you to band-pass filter and to rectify or square the signals.
% cf. Pfurtscheller, Da Silva: ERD/ERS basic priciples
%
% IN   epo     - data structure of epoched data
%      refIval - reference interval  [start msec, end msec]
%      smoothy - length of smoothing interval in msec, default no smoothing
%
% OUT  epo     - updated data structure
%
% SEE proc_filtfilt, proc_squareChannels, proc_rectifyChannels

% bb, ida.first.fhg.de


T= size(dat.x, 1);
riv= getIvalIndices(refIval, dat);
ref= repmat(mean(dat.x(riv,:,:)), [T 1 1]);

%dat.x= 100*(dat.x - ref)./ref;
dat.x= 100*(dat.x./ref - 1);
dat.refIval= refIval;

if exist('smoothy','var') & ~isempty(smoothy),
  dat= proc_movingAverage(dat, smoothy);
end

if isfield(dat, 'title'),
  dat.title= ['ERD: ' dat.title];
end
dat.yUnit= '%';
