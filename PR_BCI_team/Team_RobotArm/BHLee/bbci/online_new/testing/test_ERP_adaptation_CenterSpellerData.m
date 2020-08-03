%% params

BC= [];
BC.fcn= @bbci_calibrate_ERP_Speller;

BC.settings.nClasses= 6;
BC.settings.model = {'RLDAshrink', 'store_means', 1, 'store_cov', 1};

BC.folder= fullfile(EEG_RAW_DIR, 'VPibq_11_05_18');
BC.file= 'calibration_CenterSpellerFixedSequenceVPibq';
BC.read_param= {'fs',100};
BC.marker_fcn= @mrk_defineClasses;
BC.marker_param= {{[31:49], [11:29]; 'target', 'nontarget'}};


% In demos, we just write to the temp folder. Otherwise, the default
% choice would be fine.
BC.save.folder= TMP_DIR;
BC.log.folder= TMP_DIR;

bbci= struct('calibrate', BC);

%% calibration
[bbci, calib]= bbci_calibrate(bbci);
bbci_save(bbci);

%% insert segment markers

mrk_old = calib.mrk;
% mrk_old = mrk_selectEvents(mrk_old, 1:60); % for testing

% adapt after every classifier decision, i.e. after a full stimulation
% sequence has ended -> end_segment markers
end_segment_marker = 181;
min_diff_time = 1000; % in ms
diff_times = diff(mrk_old.time);
% figure, plot(diff_times);
pause_idx = find(diff_times>min_diff_time);
start_trial_idx = [1, pause_idx+1];
end_trial_idx = [pause_idx length(mrk_old.pos)];
n_trials = length(start_trial_idx);
n_events = length(mrk_old.pos);

% create new mrk
mrk_adapt = mrk_old;
mrk_adapt.pos = zeros(1,n_events+2*n_trials);
mrk_adapt.toe = zeros(1,n_events+2*n_trials);
mrk_adapt.time = zeros(1,n_events+2*n_trials);
mrk_adapt.y = zeros(3,n_events+2*n_trials);
mrk_adapt.className = [mrk_adapt.className {'end_segment'}];
idx = 1;
for k=1:n_trials
    
    % insert stimulus markers within this segment
    sub_trial_idx_old = start_trial_idx(k):end_trial_idx(k);
    sub_trial_idx_new = (idx+1):(idx+length(sub_trial_idx_old));
    mrk_adapt.pos(sub_trial_idx_new) = mrk_old.pos(sub_trial_idx_old);
    mrk_adapt.time(sub_trial_idx_new) = mrk_old.time(sub_trial_idx_old);
    mrk_adapt.toe(sub_trial_idx_new) = mrk_old.toe(sub_trial_idx_old);
    mrk_adapt.y(1:2,sub_trial_idx_new) = mrk_old.y(:,sub_trial_idx_old);
    
    idx = idx + length(sub_trial_idx_new) + 1;
    
    % insert end_segment marker
    mrk_adapt.pos(idx) = mrk_old.pos(end_trial_idx(k)) + 100;
    mrk_adapt.time(idx) = mrk_old.time(end_trial_idx(k)) + 1000;
    mrk_adapt.toe(idx) = end_segment_marker;
    mrk_adapt.y(3,idx) = 1;
    
    idx = idx + 1;
end
    


%% adaptation parameters

bbci.adaptation.active= 1;
bbci.adaptation.fcn= @bbci_adaptation_pcov_ERP;
bbci.adaptation.load_classifier= 0; % no need to load classifier

opt_adapt = [];
opt_adapt.alpha = 0.05;
opt_adapt.mrk_end_of_segment = end_segment_marker;
opt_adapt.min_n_data_points = 30;
opt_adapt.mrk_stimuli = bbci.control.condition.marker; % this is ugly! But I don't now yet
        % how to get the stim markers from within the adaptation
        % function... Therefore one has to set them by hand.

bbci.adaptation.param= {opt_adapt};

% bbci.adaptation.filename= fullfile(bbci.calibrate.save.folder, bbci.calibrate.save.file);
% bbci.adaptation.log.output= 'screen';

%% simulated online mode
% test consistency of classifier outputs in simulated online mode

bbci.source.acquire_fcn= @bbci_acquire_offline;
test_mrk = mrk_selectEvents(mrk_adapt, 1:70);
bbci.source.acquire_param= {calib.cnt, test_mrk};

bbci.log.output= 'screen&file';
bbci.log.folder= TMP_DIR;
bbci.log.classifier= 1;

data= bbci_apply_uni(bbci);


log_format= '%fs | M(%u) | %fs | [%f] | %s';
[time, marker_desc, marker_time, cfy, control]= ...
    textread(data.log.filename, log_format, ...
             'delimiter','','commentstyle','shell');

isequal(marker_desc, test_mrk.toe')

ref_ival= bbci.feature.proc{1}{2};
cfy_ival= bbci.feature.proc{2}{2};
epo= cntToEpo(calib.cnt, test_mrk, bbci.feature.ival, 'clab', bbci.signal.clab);
fv= proc_baseline(epo, ref_ival, 'beginning_exact');
fv= proc_jumpingMeans(fv, cfy_ival);
out= applyClassifier(fv, 'LDA', bbci.classifier.C);

% validate classifier outputs of simulated online and offline processing
max(out(:)- cfy)
