%% newblock
bvr_sendcommand('checkimpedances');
fprintf('\nPrepare cap. Press <RETURN> when finished.\n');
pause

% %% newblock - Relax measurement
% % TODO: to decide whether to do it or not
% fprintf('\n\nRelax recording.\n');
% [seq, wav, opt]= setup_season13_relax;
% fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
% pause
% stim_artifactMeasurement(seq, wav, opt);
% fprintf('Press <RETURN> when ready to go to the FEEDBACK runs.\n');
% pause
% close all
% 
cfy_name= 'bbci_classifier_carryover_32ch.mat';

dd= dir([EEG_RAW_DIR VP_CODE '*']);
% 
% if length(dd) > 1
%   warning('More session with the same subject code are present. Taking the last classifier.')
%   for idd= length(dd):-1:1
%     if exist([dd(idd).name cfy_name], 'file')
%       copyfile([dd(end).name cfy_name], TODAY_DIR);
%       break;
%     end
%   end
% end

load([TODAY_DIR cfy_name]);
uc = 2.^(-[10:5:80]./8);
switch VP_CAT 
  case 1
    iUC_mean= 7;
    iUC_pcov= 8;
  case 2
    iUC_mean= 6;
    iUC_pcov= 7;
  case 3
    iUC_mean= 5;
    iUC_pcov= 6;
end

CLSTAG= [upper(bbci.classes{1}(1)) upper(bbci.classes{2}(1))];

clidx1= find(CLSTAG(1)=='LRF');
clidx2= find(CLSTAG(2)=='LRF');
  
%% newblock - Runs 1, 30 trials per class
% TODO: write the explanation in english
desc= stimutil_readDescription(['season13_imag_fbarrow_' CLSTAG]);

stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to start feedback: ');

fb.classes= bbci.classes;
fb.classesMarkers= [int16(clidx1) int16(clidx2)];
fb.trialsPerClass = int16(32);
fb.pauseAfter= int16(16);

% bbci.feature.proc= {bbci.cont_proc.proc{1}, bbci.feature.proc{:}};
% bbci.cont_proc.proc= {bbci.cont_proc.proc{2}};
bbci.adaptation.fcn= @bbci_adaptation_pcovmean;
bbci.adaptation.param= {struct('ival', bbci.analyze.ival, 'UC_mean', uc(iUC_mean),'UC_pcov', uc(iUC_pcov),'mrk_start', [clidx1 clidx2])};
bbci.adaptation.filename= '$TMP_DIR/bbci_classifier_cspp_csp32_pcovmean';

%% pyff
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin',0, 'gui',0);
pause(4)

% % bbci= merge_structs(bbci_cfy, bbci_default);
% bbci.adaptation= bbci_default.adaptation;
% bbci.adaptation.fcn= @bbci_adaptation_cspp;
% bbci.adaptation.param= {struct('ival', bbci.analyze.ival, 'UC', uc(iuc), 'mrk_start', [clidx1 clidx2])};
% bbci.adaptation.param= {struct('ival', bbci.analyze.ival, 'featbuffer',
% fv, 'mrk_start', [clidx1 clidx2])};
% bbci.adaptation.filename= '$TMP_DIR/bbci_classifier_cspp_csp32_retrain';
% bbci.adaptation.mode= 'classifier';

% bbci.adaptation.param= {struct('ival', bbci.analyze.ival, 'featbuffer', bbci.analyze.features, 'mrk_start', [clidx1 clidx2])};

for irun= 1:3    
      
  pyff('init','FeedbackCursorArrow3');
  pause(3)

  pyff('set', fb);
  pyff('save_settings', [pyff_fb_setup '_pcovmean']);

  %% run
  pause(2);
  pyff('play','basename', 'imag_fbarrow_cspp_pcovmean');
  bbci_apply(bbci);

  pause(1);
  pyff('stop');
  fprintf('close the pyff console');
  pyff('quit');
  keyboard
  
end

fprintf('Close the pyff console');

bbci.setup_opts= [];
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_cspp_*');
bbci.setup_opts.model= {'RLDAshrink', 'scaling', 1};
bbci.setup_opts.band= 'auto';
bbci.setup_opts.ival= 'auto';
bbci.setup_opts.patch= 'auto';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspp_auto_run123');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all

bbci_bet_finish;

%% newblock, Run 4 - Hexospell without adaptation

% TODO: is there any parameter to set? Probably the screen position, to be
% implemented

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin',0, 'gui',0);
pause(4)
pyff('init','HexoSpeller');

%% test, no basename, so the data will not be recorded
pause(1);
pyff('play')
bbci_apply(bbci);

% save the settings with subject dependent design
% pyff('save_settings', [pyff_fb_setup '_MI_HexoSpeller']):

% you have to stop the feedback manually

% pyff('stop');
% pyff('quit'); 


%% run
pyff('init','HexoSpeller');
% pause(2)
% pyff('set',fb);
pyff('save_settings', [pyff_fb_setup '_hexospeller']);

pause(1);
pyff('play','basename', 'hexospeller_cspp');
bbci_apply(bbci);

pause(1);
bbci_acquire_bv('close');
pyff('stop');
pyff('quit'); 