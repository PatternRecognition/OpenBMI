function bbci= bbci_calibrate_compatibility(bbci, name)
%BBCI_CALIBRATE_COMPATIBILTY - Wrapper to use the old bbci_bet_analyze scripts
%
%Synopsis:
%  BBCI= bbci_calibrate_compatibility(BBCI, NAME)
%
%To get a description on the structures 'bbci', type
%help bbci_calibrate_structures

% 09-2011 Benjamin Blankertz

global TMP_DIR

BD= bbci.data;
Cnt= BD.cnt;
Cnt.short_title= 'XYZ';
mrk= BD.mrk;
mnt= BD.mnt;

bbci_memo= bbci;
bbci= rmfield(bbci, 'data');

bbci.setup_opts= bbci.calibrate.settings;
bbci.setup= bbci.calibrate.settings.type;
bbci.withgraphics= 1;
bbci_memo.data_reloaded= 1;
bbci_bet_analyze;
bbci.save_name= [TMP_DIR '/bbci_classifier_old_format'];
bbci_bet_finish

bbci= bbci_memo;
bbci.calibrate.settings= bbci_bet.setup_opts;
bbci_old= load(bbci.save_name);
bbci_new= bbci_apply_convertSettings(bbci_old);
bbci=