fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

%-newblock Augen geschlossen entfernen!
block_no=1;  
for set_no= 1:2;
  for word_no= 1:2;
    for speaker_no= 1:2;
      setup_bfnt_a3_season2_calibration_set;
      fprintf('Press <RETURN> to start calibration #%d of BFNT-A3 season 2.\n', block_no);
      pause; fprintf('Ok, starting...\n');
      stim_oddballAuditory(N, opt);
      block_no= block_no+1;
    end
  end
end
analyze_mmn_calibration_bfnt_a3_season2;
% Für Pascal: selected_set=[660 885 1265 1425] eigegeben!

%-artifactMeasurement
[seq, wav, opt]= setup_season10_artifacts('clstag', '');
fprintf('Press <RETURN> to start TEST artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt, 'test',1);
fprintf('Press <RETURN> to start artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> to start RELAX measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%-newblock
clf;
stimutil_fixationCross;
block_no=1; 
for run= 1:4;
 	 for word_no= 1:2;
      for speaker_no= 1:2;
  			setup_bfnt_a3_season2_oddballAuditory;
  			fprintf('Press <RETURN> to start test #%d of BFNT-A3 season 2.\n', block_no);
  			pause; fprintf('Ok, starting...\n');
  			stim_oddballAuditory(N, opt);
        block_no= block_no+1;
      end  
    end
end



%% Convert EEG Data to Matlab format
cd([BCI_DIR 'investigation\projects\project_bfnt_a3_season2']);
convert_today;

