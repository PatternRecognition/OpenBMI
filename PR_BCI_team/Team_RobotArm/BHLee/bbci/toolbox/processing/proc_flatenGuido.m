function dat2= proc_flatenGuido(dat, b, a)
%dat= proc_flatenGuido(dat)
%
% reshape data matrix to data vector (clash all but last dimensions)

% bb, ida.first.fhg.de
% UPDATE for featureCombination by DUDU, 03.07.2002

if ~isstruct(dat) & ~iscell(dat)
  sz= size(dat);
  dat2= reshape(dat, [prod(sz(1:end-1)) sz(end)]); 
  return
end

if iscell(dat)
  for j = 1:length(dat)
    dat2{j} = proc_flatenGuido(dat{j});
  end
  return
end

dat2 = copyStruct(dat,'x');
dat2.x = proc_flatenGuido(dat.x);

