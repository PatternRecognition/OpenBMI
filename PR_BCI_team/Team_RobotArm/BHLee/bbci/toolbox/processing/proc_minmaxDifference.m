function dat= proc_minmaxDifference(dat)
%

sx= size(dat.x);
xx= max(dat.x(:,:),[],1) - min(dat.x(:,:),[],1);
dat.x= reshape(xx, [1 sx(2:end)]);

if isfield(dat, 't'),
  dat= rmfield(dat, 't');
end
