% opt = struct('subdir','Matthias_06_10_16/');
% opt.fs = 100;
% opt.nChans = 12;

% eegfile_onlinesaveMatlab('init',opt);

% A = randn(12,100);
% tic
% for ii = 1:1000
%   eegfile_onlinesaveMatlab('store',A,[1 10],[1 2]);
% end
% eegfile_onlinesaveMatlab('close');
% toc
% %profile report
  
  
% % reconstruct the online saved eeg:
% filename = '/home/neuro/data/BCI/bbciRaw/Matthias_06_10_16/eeg_Matthias_001';
% [cnt,mrk] = eegfile_onlineloadMatlab('/home/neuro/data/BCI/bbciRaw/Matthias_06_10_16/eeg_Matthias_001');

opt = struct('subdir','Matthias_06_10_16/');
opt.fs = 100;
opt.nChans = 12;
clab = {};
for ii = 1:opt.nChans
  clab{ii} = ['Ch' num2str(ii)];
end
state = struct('x',[]);
state.clab = clab;

eegfile_onlinesaveBV('init',opt,state);

A = randn(12,100);
tic
for ii = 1:1000
  eegfile_onlinesaveBV('store',A,A,[1 10],{'R1' 'S2'});
end
eegfile_onlinesaveBV('close');
toc
%profile report
  
  
% reconstruct the online saved eeg:
file = '/home/neuro/data/BCI/bbciRaw/Matthias_06_10_16/eeg_Matthias_001';
[cnt,mrk] = eegfile_loadBV(file);