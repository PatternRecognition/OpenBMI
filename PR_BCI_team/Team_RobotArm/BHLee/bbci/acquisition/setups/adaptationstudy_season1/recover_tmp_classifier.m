function recover_tmp_classifier(TODAY_DIR, CLSTAG, policy)

bbci.adaptation.policy= policy;

global EEG_RAW_DIR TMP_DIR

today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
tmpfile= [TMP_DIR 'adaptation_' bbci.adaptation.policy '_' today_str];

switch(bbci.adaptation.policy),
 case 'pcovmean',
  setupfile= [EEG_RAW_DIR ...
              'subject_independent_classifiers/Lap_C3z4_bp_' CLSTAG] ;
 case 'lapcsp',
  setupfile= [TODAY_DIR 'bbci_classifier_lapcsp_setup_001'];
 case 'pmean',
  setupfile= [TODAY_DIR 'bbci_classifier_cspauto_setup_001'];
 otherwise,
  error('policy unknown');
end

S= load(setupfile);
T= load(tmpfile);

%% replace initial classifier with the adapted one
S.cls= T.cls;

save([setupfile '_' today_str], '-STRUCT','S');
