EEG_IMPORT_DIR = 'I:\EEG_Import\';
EEG_EXPORT_DIR = 'I:\EEG_Daten\';
% read the data.
%filename = 'Gabriel_06_10_18\VEP_curio_6Hz_60s.txt';
filename = 'Gabriel_06_10_18\VEP_curio_2Hz_180s.txt';
file = [EEG_IMPORT_DIR filename];
fid = fopen(file,'r');

cnt = struct;
cnt.clab = {'Trigger','O1','O2','CP4','FC3','FC4','CP3','Cz'};%strread(fgets(fid),'%s');
fclose(fid);
cnt.title = [filename(1:end-4) '_BV'];
cnt.fs = 1000;
cnt.x = textread(file,'%n','headerlines',1);
cnt.x = reshape(cnt.x,[length(cnt.clab) prod(size(cnt.x))/length(cnt.clab)]);
cnt.x = cnt.x';

% construct the markers from the first channel.
mrk = struct;
mrk.fs = 1000;

mrk_high = find(diff(cnt.x(:,1))>.5);
mrk_high(find(diff(mrk_high)==1)) = [];
  
mrk_low = find(diff(cnt.x(:,1))<-.5);
mrk_low(find(diff(mrk_low)==1)) = [];

[mrk.pos,toe_ind] = sort([mrk_high' mrk_low']);
mrk.toe = [ones(1,length(mrk_high)) 2*ones(1,length(mrk_low))];
mrk.toe = mrk.toe(toe_ind);

classDef = {1 2;'High','Low'};
mrk = makeClassMarkers(mrk,classDef);

%epo = makeEpochs(cnt,mrk,[-200 1500]);
%epo_av = proc_average(epo);
eegfile_writeBV(cnt,[],'float32');