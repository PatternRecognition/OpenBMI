
%execute setup_musical_tension first


%-newblock
%% relax recording
%global VP_SCREEN
%VP_SCREEN=[0 0 1000 800]
%-newblock
fprintf('\nArtifact recording.\n');
opt_art=struct;
opt_art.language='german';
[seq, wav, opt]= setup_musical_tension_artifacts(opt_art);

%-newblock
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_musical_tension_relax(opt_art);

%-newblock
% music presentation 1
opt_music=struct;
opt_music.joystick=0;
opt_music.show_image=1;
opt_music.playlist='playlist1_wav.txt';
opt_music.mssg='musical_tension_block1';
%for testing:
%opt_music.bv_host=[];
%opt_music.bbv_host=[];

[opt]= setup_music_presentation(opt_music); % hp:veraendern

%-newblock
% tension ratings practice
opt_practice=struct;
opt_practice.joystick=0;
opt_practice.show_image=0;
opt_practice.countdown=3;
opt_practice.test=1;
%for testing
%opt_practice.bv_host=[];
%opt_practice.bbv_host=[];
%opt_practice.mssg=[];

%practice run 1
opt_practice.mssg='musical_tension_block2';
opt_practice.playlist='playlist_practice_Bach.txt';
[opt]= setup_music_presentation(opt_practice);

%practice run 2
opt_practice.mssg='';
opt_practice.playlist='playlist_practice_Tschaikovsky.txt';
[opt]= setup_music_presentation(opt_practice);

%practice run 3
opt_practice.playlist='playlist_practice_Chopin.txt';
[opt]= setup_music_presentation(opt_practice);


%-newblock
% music presentation 2: with tension ratings
opt_music.playlist='playlist2_wav.txt';
opt_music.mssg='';
opt_music.joystick=1;
opt_music.show_image=0;

%for testing:
%opt_music.bv_host=[];
%opt_music.bbv_host=[];

[opt]= setup_music_presentation(opt_music);

%-newblock
%joystick control recording
opt_joystick=struct;
%opt_joystick.test=1;
%opt_joystick.bv_host=[];
record_joystick_movements(300,opt_joystick)

%-newblock
% standard ERP recording
setup_musical_tension_standard_erp;

%for testing:

fprintf('\n\nPress <RETURN> to start Standard MMN recording\n');
pause; fprintf('Ok, starting...\n');
stim_oddballAuditory(N, opt);


%MiMu ERP recording
%****Lautst?rke runter
setup_probe_tone;
sequence_file='mimu_sequence_new';
sequ_mat= load(sequence_file);
opt.predefined_probe_tones=sequ_mat.pt_sequence;
[order, sounds_key]=load_mimu(opt);
 %opt.bv_host=[];
 %opt.bbv_host=[];
 
 fprintf('Press <RETURN> to start ProbeTone.\n');
 pause; fprintf('Ok, starting...\n');
 stim_probe_tone(order,sounds_key,opt,'test',1)

%****Lautst?rke rauf!

%-newblock
% music presentation 3
opt_music=struct;
opt_music.joystick=0;
opt_music.show_image=1;
opt_music.playlist='playlist3_wav.txt';
opt_music.mssg='';
%for testing:
%opt_music.bv_host=[];
%opt_music.bbv_host=[];

[opt]= setup_music_presentation(opt_music);


%-newblock
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_musical_tension_relax(opt_art);
