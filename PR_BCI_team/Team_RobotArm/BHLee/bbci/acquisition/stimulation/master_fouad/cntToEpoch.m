function [trial] = cntToEpoch(chan,mrk_pos,sec,fs)

length_trial = sec*fs;

for ii=1:length(mrk_pos)

trial(:,ii) = chan(mrk_pos(ii):(mrk_pos(ii)+length_trial)-1);

end


