% exec bbci/toolbox/startup/startup_bbci.m
% exec bbci/toolbox/startup/startup_new_bbci_online.m
% exec send_udp_xml('init', host, port) to set up connection to pyff
% port = 12345

addpath('../')

BC= [];
BC.fcn= @bbci_calibrate_csp;
% BC.folder= EEG_RAW_DIR;
BC.file= 'VPkg_08_08_07/imag_arrowVPkg';
BC.read_param= {'fs',100};
BC.marker_fcn= @mrk_defineClasses;
BC.marker_param= {{1, 2, 3; 'left', 'right', 'foot'}};

% In demos, we just write to the temp folder. Otherwise, the default
% choice would be fine.
TMP_DIR= '.'; 
BC.save.folder= TMP_DIR;
BC.log.folder= TMP_DIR;

bbci= struct('calibrate', BC);

[bbci, calib]= bbci_calibrate(bbci);
% bbci_save(bbci, calib);

bbci.feedback.receiver = 'pyff';

% test consistency of classifier outputs in simulated online mode
bbci.source.acquire_fcn= @bbci_acquire_offline;
bbci.source.acquire_param= {calib.cnt, calib.mrk};

bbci.log.output= 'screen&file';
bbci.log.folder= TMP_DIR;
bbci.log.classifier= 1;

data= bbci_apply(bbci);

log_format= '%fs | [%f] | {cl_output=%f}';
[time, cfy, ctrl]= textread(data.log.filename, log_format, ...
                            'delimiter','','commentstyle','shell');

cnt_cfy= struct('fs',25, 'x',cfy, 'clab',{{'cfy'}});
mrk_cfy= mrk_selectClasses(calib.mrk, calib.result.classes);
mrk_cfy= mrk_resample(mrk_cfy, cnt_cfy.fs);
epo_cfy= cntToEpo(cnt_cfy, mrk_cfy, [0 5000]);
fig_set(1, 'name','classifier output'); clf;
plotChannel(epo_cfy, 1);
