function dat= proc_power2dB(dat);
%dat= proc_power2dB(dat);

dat.x= 10 * log10(dat.x);
dat.yUnit= 'dB';
