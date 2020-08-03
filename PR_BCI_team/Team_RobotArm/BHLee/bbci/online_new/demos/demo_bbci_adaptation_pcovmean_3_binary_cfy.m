% Subject-independent kickstart classifier
cfy_dir= [EEG_RAW_DIR 'subject_independent_classifiers/vitalbci_season2/'];
bbci= load([cfy_dir 'kickstart_vitalbci_season2_C3CzC4_9-15_15-35']);

% EEG file used of offline simulation of online processing
eeg_file= 'VPkg_08_08_07/imag_arrowVPkg';
[cnt, mrk]= eegfile_loadMatlab(eeg_file);

% Specification for pseudo-online analysis
bbci.source.acquire_fcn= @bbci_acquire_offline;
bbci.source.acquire_param= {cnt, mrk};

bbci.log.output= 'file';
bbci.log.folder= TMP_DIR;
bbci.log.classifier= 1;

data= bbci_apply(bbci);


%% Evaluation: extract classifier outputs from log-file and
%   show traces of event-related classifier outputs for each combination

log_format= '%fs %s %s';
[time, cfystr, ctrlstr]= ...
    textread(data.log.filename, log_format, ...
             'delimiter','|','commentstyle','shell');
cfy= cell2mat(cellfun(@str2num, cfystr, 'UniformOutput',0));

cnt_cfy= struct('fs',25, 'x',cfy, 'clab',{{'cfy-LR','cfy-LF','cfy-FR'}});
mrk_cfy= mrk_resample(mrk, cnt_cfy.fs);
epo_cfy= cntToEpo(cnt_cfy, mrk_cfy, [0 5000]);
fig_set(1, 'name','classifier outputs'); clf;
grid_plot(epo_cfy);
