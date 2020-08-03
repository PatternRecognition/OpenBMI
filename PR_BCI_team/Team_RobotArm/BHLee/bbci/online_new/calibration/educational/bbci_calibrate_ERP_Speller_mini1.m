function [bbci, data]= bbci_calibrate_ERP_Speller_mini1(bbci, data)
%BBCI_CALIBRATE_ERP_SPELLER_MINI1 - Calibrate for ERP Spellers - Short version
%
%This version is for educational purpose. It shows what is absolutely
%essential for setting up a BBCI classifier for ERP Speller classification.
%
%Synopsis:
% [BBCI, DATA]= bbci_calibate_ERP_Speller_mini1(BBCI, DATA)
% 
%Arguments:
%  BBCI -  the field 'calibrate' holds parameters specific to ERP Spellers.
%  DATA -  holds the calibration data
%  
%Output:
%  BBCI - Updated BBCI structure in which all necessary fields for
%     online operation are set, see bbci_apply_structures.
%  DATA - As input but added some information of the analysis that might
%     be reused in a second run

% 11-2011 Benjamin Blankertz


opt= set_defaults(bbci.calibrate.settings, ...
                  'ref_ival', [-200 0], ...
                  'cfy_ival', [100 150; 150 200; 200 250; 250 300; 300 400; 
                               400 500; 500 600; 600 700], ...
                  'cfy_clab', {'not','E*','Fp*','AF*','A*'}, ...
                  'model', 'RLDAshrink', ...
                  'nSequences', 5, ...
                  'nClasses', 6, ...
                  'cue_markers', [11:16,21:26,31:36,41:46]);

bbci.signal.clab= data.cnt.clab(chanind(data.cnt, opt.cfy_clab));

bbci.feature.proc= {{@proc_baseline, opt.ref_ival, 'beginning_exact'}, ...
                    {@proc_jumpingMeans, opt.cfy_ival}};
bbci.feature.ival= [opt.ref_ival(1) opt.cfy_ival(end)];

fv= cntToEpo(data.cnt, data.mrk, bbci.feature.ival, ...
             'clab',bbci.signal.clab);
fv= bbci_calibrate_evalFeature(fv, bbci.feature);
bbci.classifier.C= trainClassifier(fv, opt.model);

bbci.control.fcn= @bbci_control_ERP_Speller;
bbci.control.param= {copy_fields(opt, {'nClasses', 'nSequences'})};
bbci.control.condition.marker= opt.cue_markers;

bbci.quit_condition.marker= 255;

data.features= fv;
