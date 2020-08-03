
%execute setup_musical_tension_behavioral first


%-newblock
%% relax recording
%global VP_SCREEN
%VP_SCREEN=[0 0 1000 800]
%-newblock
%-newblock
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

