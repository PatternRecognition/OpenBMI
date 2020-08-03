function dat= proc_addTTDchannel(dat)
%cnt= proc_addTTDchannel(cnt)

ttd_chans= chanind(dat, 'Cz', 'A1', 'A2', 'EOGv');
ttd_weights= [1 -0.5 -0.5 -0.12]';

dat.x(:,end+1)= dat.x(:,ttd_chans) * ttd_weights;
dat.clab= cat(2, dat.clab, {'Feedb'});
