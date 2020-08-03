function dat= proc_addBiasComponent(dat)
%dat= proc_addBiasComponent(dat)

sz= size(dat.x);
dat.x= cat(1, dat.x, -ones([1 sz(2:end)]));
