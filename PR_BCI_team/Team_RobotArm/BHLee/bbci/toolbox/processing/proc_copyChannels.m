function dat= proc_copyChannels(dat, dat2, copyChans)
%dat= proc_copyChannels(dat, dat2, copyChans)
%
% appends channels 'copyChans' of dat2 to dat
% format of 'copyChans' as in function chanind
%
% SEE chanind

% bb, ida.first.fhg.de


copyChans= chanind(dat2, copyChans);

if ~isempty(copyChans),
  dat.x= cat(2, dat.x, dat2.x(:,copyChans,:));
  dat.clab= {dat.clab{:}, dat2.clab{copyChans}};
end
