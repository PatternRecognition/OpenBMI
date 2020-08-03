% [dat, b, a]=proc_filtButter(dat, order, band)
function [dat,b,a]=proc_filtButter(dat, order, band)
if order==5 & isequal(band,[7 30]) & dat.fs==100
  load 'butter730.mat'
else
  [b,a]=butter(order, band/dat.fs*2);
end

dat = proc_filt(dat, b, a);
if length(band)==2
  dat.title = sprintf('%s [%.1f %.1f]', dat.title, band(1), band(2));
else
  dat.title = sprintf('%s [0 %.1f]', dat.title, band);
end

