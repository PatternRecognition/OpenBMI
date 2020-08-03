% Stimulus selection BFNT A3, season2

% 4 different runs used per experiment, consisting of 4 blocks each
%(Wort1_Speaker1, Wort2_speaker2, Wort2_speaker1, Wort1_speaker2) 
%(Wort2_speaker2, Wort1_speaker1, Wort1_speaker2, Wort2_speaker1) 
%(Wort1_speaker2, Wort2_speaker1, Wort2_speaker2, Wort1_speaker1) 
%(Wort2_speaker1, Wort1_speaker2, Wort1_speaker1, Wort2_speaker2)

% left column: word_no, right column: speaker_no
runA= [1 1; 2 2; 2 1; 1 2];
runB= [2 2; 1 1; 1 2; 2 1];
runC= [1 2; 2 1; 2 2; 1 1];
runD= [2 1; 1 2; 1 1; 2 2];

% note: 
% switch between 'beginning speakers' between runs: 
% 4x2x1x1 possible combinations, i.e. exp1-8
exp1= [runA; runC; runD; runB];
exp2= [runA; runB; runD; runC];

exp3= [runB; runA; runC; runD];
exp4= [runB; runD; runC; runA];

exp5= [runC; runA; runB; runD];
exp6= [runC; runD; runB; runA];

exp7= [runD; runB; runA; runC];
exp8= [runD; runC; runA; runB];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select experiment:
exp= exp8; 
% Merlin: exp4;
% Toralf: exp5;
% Nathalie: exp6;
% Lina: exp7;
% Sebastian: exp8;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

%-newblock Augen geschlossen entfernen!
% Calibration set
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
% Für Pascal: selected_set=[660 885 1265 1425] eingegeben!
% für David: selected_set=[660 885 1265 1425]
% für Miriam: selected_set=[885 1265 1425 1585]
% für Merlin: selected_set=[1585 1825 1985 2305]
% für Toralf: selected_set=[885 1265 1425 1585]
% A1 und A2(Mastoid Elektroden sind falsch zugeordet: auf dem Setup ist A1
% links und A2 rechts; in Realität ist aber A1 rechts und A2 links! Vor der
% Auswertung noch Tauschen! JNA
% für Nathalie: selected_set=[885 1265 1425 1585]
% für Lina: selected_set=[660 885 1265 1425]
% für Sebastian: selected_set=[660 885 1265 1425]



%-artifactMeasurement
%[seq, wav, opt]= setup_season10_artifacts('clstag', '');
%fprintf('Press <RETURN> to start TEST artifact measurement.\n');
%pause; fprintf('Ok, starting...\n');
%stim_artifactMeasurement(seq, wav, opt, 'test',1);
%fprintf('Press <RETURN> to start artifact measurement.\n');
%pause; fprintf('Ok, starting...\n');
%stim_artifactMeasurement(seq, wav, opt);
%fprintf('\n\nRelax recording.\n');
%[seq, wav, opt]= setup_season10_relax;
%fprintf('Press <RETURN> to start RELAX measurement.\n');
%pause; fprintf('Ok, starting...\n');
%stim_artifactMeasurement(seq, wav, opt);

% Versuchsanordnung 2010-11-11
%exp_test= [1 1; 1 1; 1 1; 1 2; 2 1; 2 2; 2 1; 2 2; 1 2; 2 2; 1 2; 2 1; 1 1; 2 2; 2 1; 1 2]

%-newblock
clf;
stimutil_fixationCross;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for block_no= 1:length(exp),
  word_no=  exp(block_no,1);
  speaker_no= exp(block_no,2);
  % do experiment with the designated stimulus material
  setup_bfnt_a3_season2_oddballAuditory;
  fprintf('Press <RETURN> to start test #%d of BFNT-A3 season 2.\n', block_no);
  pause; fprintf('Ok, starting...\n');
  stim_oddballAuditory(N, opt);
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Convert EEG Data to Matlab format
%cd([BCI_DIR 'investigation\projects\project_bfnt_a3_season2']);
%convert_today;

