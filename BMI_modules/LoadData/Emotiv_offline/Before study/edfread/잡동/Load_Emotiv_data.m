function [EEG]= Load_Emotiv_data(xEEG)

EEG.data.x=xEEG.data';

EEG.data.fs=xEEG.srate';
EEG.data.nCh=xEEG.nbchan;
% EEG.data.chSet=3;
for i=1:EEG.data.nCh
  EEG.data.chSet{1,i}=xEEG.chanlocs(i,1).labels;
end

