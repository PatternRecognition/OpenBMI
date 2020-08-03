clab= {'F3','Fz','F4', 'C3','Cz','C4', 'P3','Pz','P4'};
cfy_ival= [90 110; 110 150; 150 250; 250 400; 400 750];
% Generate random classifier of correct format
C= struct('b',0);
C.w= randn(length(clab)*size(cfy_ival,1), 1);

bbci= struct;
bbci.source.acquire_fcn= @bbci_acquire_bv;
bbci.source.acquire_param= {struct('fs', 100)};
bbci.source.marker_mapping_fcn= [];

bbci.cont_proc.clab= clab;

bbci.feature.proc= {{@proc_baseline, [-200 0]}, ...
                    {@proc_jumpingMeans, cfy_ival}};
bbci.feature.ival= [-200 750];

bbci.classifier.C= C;

bbci.control.fcn= @bbci_control_ERP_Speller_binary;
bbci.control.param= {struct('nClasses',6, 'nSequences',10)};
bbci.control.condition.marker= [11:16,21:26,31:36,41:46];

bbci.feedback.receiver = 'pyff';

bbci.quit_condition.marker= 255;

bbci.log.output= 'screen';
bbci.log.classifier= 1;

pyff('quit');
bbci_acquire_bv('close')

setup_ERP_speller
pyff('setint','offline',0);
pyff('set','desired_phrase','');
pyff('play');
bbci_apply(bbci);
