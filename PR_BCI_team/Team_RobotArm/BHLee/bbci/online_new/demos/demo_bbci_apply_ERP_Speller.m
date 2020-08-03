eeg_file= 'VPiac_10_10_13/CenterSpellerMVEP_VPiac';
[cnt, mrk]= eegfile_loadMatlab(eeg_file, 'vars',{'cnt','mrk'});

clab= {'F3','Fz','F4', 'C3','Cz','C4', 'P3','Pz','P4'};
ref_ival= [-200 0];
cfy_ival= [90 110; 110 150; 150 250; 250 400; 400 750];
% Generate random classifier of correct format
C= struct('b',0);
C.w= randn(length(clab)*size(cfy_ival,1), 1);

bbci= struct;
bbci.source.acquire_fcn= @bbci_acquire_offline;
bbci.source.acquire_param= {cnt, mrk};

bbci.signal.clab= clab;

bbci.feature.proc= {{@proc_baseline, ref_ival}, ...
                    {@proc_jumpingMeans, cfy_ival}};
bbci.feature.ival= [ref_ival(1) cfy_ival(end)];

bbci.classifier.C= C;

bbci.control.fcn= @bbci_control_ERP_Speller;
bbci.control.param= {struct('nClasses',6, 'nSequences',10)};
bbci.control.condition.marker= [11:16,21:26,31:36,41:46];

bbci.quit_condition.marker= 255;
bbci.quit_condition.running_time= 2*60;

bbci.log.output= 'screen&file';
bbci.log.file= fullfile(TMP_DIR, 'log');
bbci.log.classifier= 1;

data= bbci_apply_uni(bbci);
% Of course, bbci_apply would do the very same.


%% validate simulated online results with offline classification
% read markers and classifier output from logfile
log_format= '%fs | M(%u) | %fs | [%f] | %s';
[time, marker_desc, marker_time, cfy, control]= ...
    textread(data.log.filename, log_format, ...
             'delimiter','','commentstyle','shell');

% validate makers that evoked calculation of control signals
isequal(marker_desc, mrk.toe(1:length(marker_desc))')

% offline processing of the data
epo= cntToEpo(cnt, mrk, [ref_ival(1) cfy_ival(end)], 'clab', bbci.signal.clab);
fv= proc_baseline(epo, ref_ival);
fv= proc_jumpingMeans(fv, cfy_ival);
out= applyClassifier(fv, 'LDA', bbci.classifier.C);

% validate classifier outputs of simulated online and offline processing
max(abs( out(1:length(cfy))' - cfy))

% extract control signals
isctrl= cellfun(@(x)(length(x)>2), control);
control_str= sprintf('%s\n', control{find(isctrl)});
[var_name, var_value]= strread(control_str, '{%s%f}', 'delimiter','=');
var_value'
