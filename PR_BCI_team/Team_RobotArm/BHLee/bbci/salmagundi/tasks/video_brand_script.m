opt= [];
opt.input_dir= [DATA_DIR 'eegVideo/favorites/'];

file= 'replay_al_04_03_29_blind';
desc={'New subject al (2nd BCI session) performs 1D cursor control', 
      'eyes closed WITHOUT getting feedback. Targets are given',
      'acoustically by the operator. The subject performs 100%',
      'of 25 trials correct (first 3 trials are for settling in).'};
%video_brandmarkBBCI(file, opt, 'desc_text',desc);
file= 'video_al_04_03_29_blind';
video_brandmarkBBCI(file, opt, 'desc_text',desc);

file= 'replay_al_05_11_07_cursor_40bpm';
desc= {'Subject al performs 1D cursor control and obtains',
       'a bitrate of above 41 bits per minute in a run',
       'of fifty trials.'};
video_brandmarkBBCI(file, opt, 'desc_text',desc);
%file= 'video_al_05_11_07_cursor_40bpm';  %% does not exist !?
%desc= {'Subject al performs 1D cursor control and obtains',
%       'a bitrate of above 41 bits per minute in a run',
%       'of fifty trials. (See replay for better display.)'};
%video_brandmarkBBCI(file, opt, 'desc_text',desc);

file= 'replay_ay_04_04_08_blind';
desc={'New subject ay (2nd BCI session) performs 1D cursor control', 
      'eyes closed WITHOUT getting feedback. Targets are given',
      'acoustically by the operator. The subject makes only one',
      'mistake in 25 trials (first 3 trials are for settling in).'};
%video_brandmarkBBCI(file, opt, 'desc_text',desc);
file= 'video_ay_04_04_08_blind';
video_brandmarkBBCI(file, opt, 'desc_text',desc);


file= 'replay_al_06_03_09_hex-o-spell_177';
desc= {'Subject al spells a phrase using the mental typewriter',
       'Hex-o-Spell. The spelling speed is above 8 chars/min',
       'including the time to correct typing errors.'};
video_brandmarkBBCI(file, opt, 'desc_text',desc);

file= 'video_av_04_03_31_cursor_1dr';
desc= {'New subject av (2nd BCI session; 1st was on a beta', 
       'version) performs 1D cursor control. The blue field',
       'indicates the target. A trial ends when the cursor',
       'hits either side. Colors indicate hit (green) or miss (red).'};
video_brandmarkBBCI(file, opt, 'desc_text',desc);

file= 'video_aw_hex-o-spell_Sonne';
desc= {'Subject aw writes a sentence using (a preliminary version of)',
       'the mental typewriter Hex-o-Spell. This sentence was one of the',
       'sentences used in the first public demonstration of',
       'the telephone invented by Philipp Reis in 1861.'};
video_brandmarkBBCI(file, opt, 'desc_text',desc);

file= 'video_al_speller';
desc= {'Subject al writes ''BBCI'' using a mental typewriter',
       'based on 1D cursor control. The alphabet is iteratively',
       'splitted in two parts. One letter is selected by a sequence',
       'of binary decisions. A backspace symbol is used for correction.'};
video_brandmarkBBCI(file, opt, 'desc_text',desc);

file= 'video_brainpong';
desc= {'Subject ay plays ''brain-pong''. The bat at the bottom',
       'of the screen is controlled  purely by brain activity',
       'induced by imagined movements.'};
video_brandmarkBBCI(file, opt, 'desc_text',desc);

file= 'video_brainpong2p';
desc= {'Subjects al and aw play ''brain-pong'' against each',
       'other. The bats are controlled purely by brain activity',
       'induced by imagined movements.'};
video_brandmarkBBCI(file, opt, 'desc_text',desc);

file= 'video_virtual_arm';
desc= {'Subject {\it aa} performs movements of the arm and the index finger.',
       'The BBCI system predicts the type of movement from the brain',
       'signals recorded before the movement starts and displays them',
       'with the virtual arm in synchrony with the movement.'};
video_brandmarkBBCI(file, opt, 'desc_text',desc);
