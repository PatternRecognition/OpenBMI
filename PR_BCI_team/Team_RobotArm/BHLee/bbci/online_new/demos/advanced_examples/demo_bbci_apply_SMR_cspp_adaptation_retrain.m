global TMP_DIR

TMP_DIR= [DATA_DIR 'results/vitalbci_season1/csp_patches/online_new/'];
d= pwd;
d= d(find(d=='/',1,'last')+1:end);
clab64 = {'F5,3,1,z,2,4,6','FFC5,3,1,z,2,4,6','FC5,3,1,z,2,4,6','CFC5,3,1,z,2,4,6','C5,3,1,z,2,4,6','CCP5,3,1,z,2,4,6','CP5,3,1,z,2,4,6', ...
  'PCP5,3,1,z,2,4,6','P5,3,1,z,2,4,6','PPO1,2','PO3,z,4'};

% EEG file used of offline simulation of online processing
% subdir= 'VPjj_08_06_10';
subdir= 'VPji_08_06_02';
sbj = subdir(1:find(subdir=='_',1,'first')-1);
eeg_file= [subdir '/imag_fbarrow' sbj];

[cnt, mrk_orig, mrk]= eegfile_loadMatlab(eeg_file, 'clab', clab64, 'vars',{'cnt','mrk_orig', 'mrk'});

% whant to cut the first 100 trials
idx_stim1= strmatch('S  1', mrk_orig.desc);
idx_stim2= strmatch('S  2', mrk_orig.desc);
idx_stim= union(idx_stim1, idx_stim2);
ival_train(1)= 1;
ival_train(2)= mrk_orig.pos(idx_stim(101)-1)*1000/cnt.fs;

bbci.clab= cnt.clab;
bbci.train_file= eeg_file;
bbci.ival= ival_train;
bbci.setup= 'cspp_auto';
bbci.setup_opts= [];
bbci.setup_opts.classes= mrk.className;
bbci.classes= mrk.className;
bbci.classDef= cat(1, {1, 2}, bbci.classes);
bbci.setup_opts.model= {'RLDAshrink', 'scaling', 1};
% TODO: try also 24 channels and see what is better
% if 48
% bbci.setup_opts.clab_csp= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,z,4'};
% if 24
bbci.setup_opts.clab_csp= {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
S=  getClassifierLog(subdir);
bbci.setup_opts.band= S.bbci.analyze.band;
bbci.setup_opts.ival= S.bbci.analyze.ival;
% bbci.setup_opts.patch= 'ten';

% bbci.setup_opts.band= 'auto';
% bbci.setup_opts.ival= 'auto';
bbci.setup_opts.patch= 'auto';
% TODO, adjust the name according to the number of channels used for csp
bbci.save_name= strcat(TMP_DIR, 'bbci_classifier_cspp_auto_csp24_retrain');
bbci.withgraphics= 1;
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish

%% in retrain the cspp change
bbci.feature.proc= {bbci.signal.proc{1}, bbci.feature.proc{:}};
bbci.signal.proc= {bbci.signal.proc{2}};

%% adaptation
bbci.adaptation.active= 1;
bbci.adaptation.fcn= @bbci_adaptation_cspp;
% if bbci.analyze.ival(2) < 2750
%   adaptation_ival= [bbci.analyze.ival(1) 2750];
% else
%   adaptation_ival= bbci.analyze.ival;
% end
adaptation_ival=[750 3750];
epo_flt= proc_filt(Cnt, bbci.analyze.filt_b, bbci.analyze.filt_a);
epo_flt= cntToEpo(epo_flt, mrk2, adaptation_ival, 'clab', bbci.analyze.clab);

bbci.adaptation.param= {struct('ival',adaptation_ival,'featbuffer', epo_flt)};
clear epo_flt
bbci.adaptation.filename= strcat(TMP_DIR, 'bbci_classifier_cspp_retrain');
bbci.adaptation.mode= 'everything_at_once';

[cnt, mrk_orig]= eegfile_readBV([eeg_file '02'],'fs', bbci.fs, 'clab', clab_load);

%740 is just to match with the 750 of the old version. Usually we put 750
%here.
bbci.feature.ival= [-740 0];
bbci.source.acquire_fcn= @bbci_acquire_offline;
bbci.source.acquire_param= {cnt, mrk_orig};

bbci.quit_condition.marker= 254;

bbci.log.output= 'screen&file';
bbci.log.file= strcat(TMP_DIR, 'log_cspp_retrain');
bbci.log.classifier= 1;
bbci.log.force_overwriting= 1;

bbci.control.fcn= @bbci_control_SMR_cursor_1d;
bbci.control.param= {struct('mrk60',1, 'mrk_start', [1 2], 'mrk_end', [11 12 21 22])};
bbci.control.condition.marker= [1,2,11,12,21,22,60];

if isequal(d, 'demos')
   % local bbci_apply_evalCondition. Needed to calculate the results.
   % Otherwise no control is needed  
  bbci.control.condition.interval= 40;
end

data= bbci_apply(bbci);

%% extract simulated online classification results
% read markers and classifier output from logfile
if isequal(d, 'demos')
  % local bbci_apply_evalCondition used. Needed to calculate the results
  results= textread(data.log.filename, '%s');    
  markers= results(3:9:end);
  idx_resp = [];
  for m= 1:length(bbci.control.condition.marker(3:end-1))    
    idx_resp= [idx_resp strmatch(['M(' num2str(bbci.control.condition.marker(3+m-1)) ')'], markers)'];
  end
  control= results(idx_resp*9);
  for trial= 1:length(control)
    idx1 = findstr(control{trial},',')+1;
    idx2 = findstr(control{trial},']')-1;
    ishit(trial)= str2num(control{trial}(idx1(end):idx2));
  end
  fprintf('%2.2f',mean(ishit)*100)  
else
  % svn bbci_apply_evalCondition used
  [time, marker_desc, marker_time, cfy_output, control]= textread(data.log.filename, log_format, ...
    'delimiter','','commentstyle','shell');
  
  % map mrk_orig.desc strings to intergers
  nEvents= length(time);
  [mrk_desc, iSR]= marker_mapping_SposRneg(mrk_orig.desc);
  % and pick those, which evoked the calculation of classifier outputs
  idx= find(ismember(mrk_desc, bbci.control.condition.marker));
  idx_control= idx(1:nEvents);
  
  % validate makers that evoked calculation of control signals
  isequal(marker_desc', mrk_desc(idx_control))
end
