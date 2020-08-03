function [cnt,mrk] = eegfile_onlineloadMatlab(filename)
% kraulem 10/06
load([filename '.mat']);
fid_dat = fopen([filename '.eeg']);
cnt=  struct('fs',opt.fs);
cnt.file = filename;
cnt.x = fread(fid_dat, [opt.nChans inf], 'double');

fid_mrk = fopen([filename '.mrk']);
mrk = struct('fs',opt.fs);
PosToe = fread(fid_mrk,[2 inf],'double');
mrk.pos = PosToe(1,:);
mrk.toe = PosToe(2,:);
