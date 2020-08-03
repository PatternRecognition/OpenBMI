pathroot='E:\svn\bbci\'
addpath([pathroot 'stimulation\sound_discretization'])
addpath([pathroot 'acquisition\setups\sound_discretization'])
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
opt_music.playlist='playlist_discr1.txt';
opt_music.mssg='sound_discretization1';
%for testing:
%opt_music.bv_host=[];
%opt_music.bbv_host=[];

[opt]= setup_sound_presentation(opt_music);


%-newblock
% music presentation 1
opt_music=struct;
opt_music.joystick=0;
opt_music.show_image=1;
opt_music.playlist='playlist_discr2.txt';
opt_music.mssg='sound_discretization1';
%for testing:
%opt_music.bv_host=[];
%opt_music.bbv_host=[];

[opt]= setup_sound_presentation(opt_music);

% hp: warum hoere ich nichts?
% answ_flag=1
% has to be testet for input
% if answ(1)
% end

%-newblock
% standard ERP recording
setup_musical_tension_standard_erp;
% hp: kein sound?

%for testing:

fprintf('\n\nPress <RETURN> to start Standard MMN recording\n');
pause; fprintf('Ok, starting...\n');
stim_oddballAuditory(N, opt);



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


%%

%-newblock
% music presentation 2: with tension ratings
opt_music.playlist='playlist_ratings.txt';
opt_music.mssg='';
opt_music.joystick=1;
opt_music.show_image=0;

[opt]= setup_sound_presentation(opt_music);

