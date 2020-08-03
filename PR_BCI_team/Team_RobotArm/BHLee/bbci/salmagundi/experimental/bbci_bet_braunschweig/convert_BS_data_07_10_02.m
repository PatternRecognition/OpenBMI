EEG_IMPORT_DIR = 'D:\EEG_Daten\';
EEG_EXPORT_DIR = 'D:\EEG_Daten\';
% read the data.

filelist= {'alpha_curio_1',...
    'alpha_curio_2',...
    'alpha_conradi_4',...
    'alpha_oehler_6',...
    'motor_curio_3',...
    'motor_conradi_5',...
    'motor_oehler_7'};
  
markertype= cat(2,repmat({'alpha'}, [1,4]),repmat({'selfpaced'},[1,3]));

for nr= 1:length(filelist),
file = ['D:\EEG_Daten\BStest_07_10_02\raw\' filelist{nr}];
[cnt,mrk,mnt]=eegfile_loadBV(file);
cnt.title= ['BStest_07_10_02/' filelist{nr}];


% construct the markers from the Trigger channel.

switch markertype{nr}
  case 'alpha'
    mrk = mrkFromTriggerChannel(cnt,'steps',[0 5.95 7.8],'trg_mrk',[1 2 0]);
    mrk= mrk_defineClasses(mrk, {1, 2; 'closed','open'});
  case 'VEP'
    mrk = mrkFromTriggerChannel(cnt,'steps',[6000 10000 0],'trg_mrk',[0 1 2]);
    classDef = {1 2;'High','Low'};
    mrk = mrk_defineClasses(mrk,classDef);
  case 'selfpaced'
    % for f and ö
    mrk = mrkFromTriggerChannel(cnt,'steps',[-4 3.6 7.8],'trg_mrk',[1 2 0]);
    mrk= mrk_defineClasses(mrk, {1, 2; 'right','left'});
    
  case 'lett'
    mrk = mrkFromTriggerChannel(cnt,'steps',[39000 4000 -27500 8500  -40000],...
      'min_len',10,'trg_mrk',0:4,'blocking_time',500);
    mrk = mrk_defineClasses(mrk,{1,2; 'left','right'});
    mrk = mrk_selectClasses(mrk,{'left','right'});
end
%
eegfile_writeBV(cnt,mrk,'float32');

%keyboard
end
