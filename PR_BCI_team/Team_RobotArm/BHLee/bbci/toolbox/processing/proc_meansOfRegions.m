function dat= proc_meansOfRegions(dat, regions)
%dat= proc_meansOfRegions(dat, regions)
%
% IN   dat     - data structure of continuous or epoched data
%      regions - cell array of intervals in which the mean should be calculated
%
% OUT  dat     - updated data structure

% bb, ida.first.fhg.de


[T, nChans, nEvents]= size(dat.x);
nSections= length(regions);

xo= zeros(nSections, nChans, nEvents);
for ir= 1:nSections,
  Ti= getIvalIndices(regions{ir}, dat);
  xo(ir,:,:)= mean(dat.x(Ti,:,:));
end
dat.x= xo;
