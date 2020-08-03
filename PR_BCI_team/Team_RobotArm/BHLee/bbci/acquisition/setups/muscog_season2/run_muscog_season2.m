bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause


%-newblock
fprintf('\nArtifact recording.\n');
[seq, wav, opt]= setup_muscog_season2_artifacts;
fprintf('Press <RETURN> when ready to start artifact measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);

%-newblock
setup_probe_tone;
fprintf('Press <RETURN> when ready to start to the ProbeTone TEST run.\n');
pause
sequence_file='mimu_sequence_1';
sequ_mat=load(sequence_file);
opt.predefined_probe_tones=sequ_mat.pt_sequence;
[order sounds_key]=load_mimu(opt);
stim_probe_tone(order,sounds_key,opt,'test',1,'howmany',8);
    

 
for l=1:5
    fprintf('Press <RETURN> when ready to start measurement.\n'); 
    pause
    sequence_file=['mimu_sequence_' num2str(l)];
    sequ_mat=load(sequence_file);
   
    opt.predefined_probe_tones=sequ_mat.pt_sequence;
    
    [order sounds_key]=load_mimu(opt);
    stim_probe_tone(order,sounds_key,opt);
             
    
end
