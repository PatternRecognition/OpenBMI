logfile = 'VPdt_06_05_08/log/feedback_speller_2d_005.log';
title = 'VPdt_06_05_08/imag_fbspeller_wordsVPdt';
fs = 1000;

l = load_feedback(logfile,fs);

mr = rmfield(l.mrk,{'counter','lognumber'});
mr.fs = fs;

mrk = struct('fs',repmat({fs},[length(mr.pos)+1,1]));

mrk(1).type = 'New Segment';
for i = 2:length(mr.pos)+1
  mrk(i).type = 'Stimulus';
end

mrk(1).desc = '';
for i = 1:length(mr.pos)
  mrk(i+1).desc = sprintf('%s% 3d',mrk(i+1).type(1),mr.toe(i));
end

offset = l.update.pos(1)-1;

mrk(1).pos = 1;
for i = 1:length(mr.pos)
  if isnan(mr.pos(i))
    mrk(i+1).pos = 1;
  else
    mrk(i+1).pos = mr.pos(i)-offset;
  end
end

for i = 1:length(mr.pos)+1
  mrk(i).length = 1;
  mrk(i).chan = 0;
end

mrk(1).time = sprintf('%d%02d%02d%02d%02d%02d000000',l.start);
for i = 1:length(mr.pos)
  mrk(i+1).time = '';
end

cnt = struct('fs',fs);
cnt.clab = {'NaN'};
cnt.x = nan*ones(mrk(end).pos+1,1);

cnt.title = [EEG_RAW_DIR title];

eegfile_writeBV(cnt,mrk,0.1);
