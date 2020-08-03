BC= [];
BC.folder= [EEG_RAW_DIR 'VPibq_11_05_18/'];
BC.file= 'calibration_CenterSpellerFixedSequenceVPibq';
BC.marker_fcn= @mrk_defineClasses;
BC.marker_param= { {[31:49], [11:29]; 'target', 'nontarget'} };
BC.save.folder= '/tmp/';
BC.save.file= 'bbci_classifier_ERP_Speller';
BC.fcn= @bbci_calibrate_ERP_Speller_v0;
BC.settings.disp_ival= [-200 800];
BC.settings.ref_ival= [-200 0];

bbci= [];
bbci.calibrate= BC;
bbci= bbci_calibrate(bbci);
bbci= bbci_save(bbci);

