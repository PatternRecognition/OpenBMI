error('obsolete');

bbci = bbci_apply_loadSettings(bbci_cfy_file);

% The following should be obsolete, since it is set in the setup file
bbci.source.marker_mapping_fcn= [];

bbci.control.fcn= @bbci_control_ERP_Speller_binary;
bbci.control.param= {struct('nClasses',6, 'nSequences',10)};
bbci.control.condition.marker= [11:16,21:26,31:36,41:46];

bbci.feedback.receiver = 'pyff';

bbci.quit_condition.marker= 255;
