bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

%-newblock
for set_no= 1:3;
  setup_bfnt_a3_season1_calibration_set;
%   if set_no==1,
%     fprintf('Press <RETURN> to TEST calibration.\n');
%     pause; fprintf('Ok, starting...\n');
%     stim_oddballAuditory(50, opt, 'test',1);
%   end
  fprintf('Press <RETURN> to start calibration #%d of BFNT-A3.\n', set_no);
  pause; fprintf('Ok, starting...\n');
  stim_oddballAuditory(N, opt);
end
analyze_mmn_calibration;

%-newblock
[seq, wav, opt]= setup_season10_artifacts('clstag', '');
fprintf('Press <RETURN> to start TEST artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt, 'test',1);
fprintf('Press <RETURN> to start artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%-newblock
stimutil_fixationCross;
setup_muscog_season2_mmn;
fprintf('\n\nPress <RETURN> to start TEST Standard MMN recording\n');
pause; fprintf('Ok, starting...\n');
stim_oddballAuditory(40, opt, 'test', 1);
fprintf('\n\nPress <RETURN> to start Standard MMN recording\n');
pause; fprintf('Ok, starting...\n');
stim_oddballAuditory(N, opt);

%-newblock
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> to start RELAX measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%-newblock
clf;
stimutil_fixationCross;
for run= 1:12,
  setup_probe_tone;
  sequence_file= ['mimu_sequence_' num2str(mod(run-1,5)+1)];
  sequ_mat= load(sequence_file);
  opt.predefined_probe_tones=sequ_mat.pt_sequence;
  [order, sounds_key]=load_mimu(opt);
  if run==1,
    fprintf('Press <RETURN> to start to the ProbeTone TEST run.\n');
    pause; fprintf('Ok, starting...\n');
    stim_probe_tone(order,sounds_key,opt, 'test',1, 'howmany',8);
  end
  fprintf('Press <RETURN> to start run #%d of ProbeTone.\n', run);
  pause; fprintf('Ok, starting...\n');
  stim_probe_tone(order,sounds_key,opt);
  
  setup_bfnt_a3_season1_oddballAuditory;
  if run==1,  
    fprintf('Press <RETURN> to start to the TEST run.\n');
    pause; fprintf('Ok, starting...\n');
    stim_oddballAuditory(60, opt, 'test',1);
  end
  fprintf('Press <RETURN> to start run #%d of BFNT-A3.\n', run);
  pause; fprintf('Ok, starting...\n');
  stim_oddballAuditory(N, opt);
   
end
%DO NOT FORGET!!!
%-newblock
stimutil_fixationCross;
setup_muscog_season2_mmn;
fprintf('\n\nPress <RETURN> to start Standard MMN recording\n');
pause; fprintf('Ok, starting...\n');
stim_oddballAuditory(N, opt);


%% Convert EEG Data to Matlab format
cd([BCI_DIR 'investigation\projects\muscog_season3']);
convert_today;
cd([BCI_DIR 'investigation\projects\project_bfnt_a3_season1']);
convert_today;
