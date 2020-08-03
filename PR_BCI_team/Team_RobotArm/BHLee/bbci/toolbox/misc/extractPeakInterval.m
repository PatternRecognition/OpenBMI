function peakIval= extractPeakInterval(epo, ival, width, chan, minormax)
%peakIval= extractPeakInterval(epo, ival, width, chan, minormax)

if ~exist('chan', 'var') | isempty(chan),
  chan= 1;
else
  chan= chanind(epo, chan);
end
if ~exist('minormax', 'var'),
  minormax= 'max';
end

iv= getIvalIndices(ival, epo);
erp= proc_classMean(epo);
switch(lower(minormax)),
 case 'min',
  [mi, iPeak]= min(erp.x(iv, chan));
 case 'max',
  [ma, iPeak]= max(erp.x(iv, chan));
 otherwise,
  error('minormax must be ''min'' or ''max''')
end

pt= erp.t(iv(1)+iPeak-1);          %% peak time
peakIval= pt + [-width/2 width/2];
