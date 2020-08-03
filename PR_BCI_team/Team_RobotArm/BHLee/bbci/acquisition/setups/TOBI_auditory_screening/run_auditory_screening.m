%% set everything up
warning('turn the volume of the MOTU to -20 and press enter to proceed');
input('');
warning('Is the proper VP_CODE set?');
input('');

global VP_SCREEN;
VP_SCREEN = [-1920 0 1920 1200];

bvr_sendcommand('loadworkspace', ['reducerbox_64std']);
    
%% short example trial of standard P300 test (ISI = 1000)
N=20;
clear opt;
setup_auditory_screening;
opt.perc_dev = 20/100;
opt.isi = 1000;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = [opt.cue_std opt.cue_std];
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = [opt.cue_dev opt.cue_dev];
opt.use_speaker = [];
%testing
opt.bv_host = '';
opt.filename = '';
opt.test = 1;
stim_oddballAuditory(N, opt);


%% do the standard P300 test (ISI = 1000)
N=250;
iterations = 2;
clear opt;
setup_auditory_screening;
opt.perc_dev = 20/100;
opt.isi = 1000;
opt.filename = 'oddballStandardMessung1000';
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = [opt.cue_std opt.cue_std];
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = [opt.cue_dev opt.cue_dev];
opt.use_speaker = [];

for i = 1:iterations,
  % do some stuff to start recording here
  % then pause
  opt.impedances = (i==1);
  stim_oddballAuditory(N, opt);
  stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');
end

%% short example trial of standard P300 test (ISI = 200)
N=20;
clear opt;
setup_auditory_screening;
opt.perc_dev = 20/100;
opt.isi = 200;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = [opt.cue_std opt.cue_std];
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = [opt.cue_dev opt.cue_dev];
opt.use_speaker = [];
%testing
opt.bv_host = '';
opt.filename = '';
opt.test = 1;
stim_oddballAuditory(N, opt);


%% do the standard P300 test (ISI = 200)
N=250;
iterations = 2;
clear opt;
setup_auditory_screening;
opt.perc_dev = 20/100;
opt.isi = 200;
opt.filename = 'oddballStandardMessung200';
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = [opt.cue_std opt.cue_std];
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = [opt.cue_dev opt.cue_dev];
opt.use_speaker = [];
opt.impedances = 0;

for i = 1:iterations,
  stim_oddballAuditory(N, opt);
  stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');
end

%% short example trial of 2-target oddball (ISI = 400)
N=30;
clear opt;
setup_auditory_screening;
opt.perc_dev = 20/100;
opt.isi = 400;
opt.use_speaker = [];
sound_dir = [BCI_DIR '\python\pyff\src\Feedbacks\Auditory_stimulus_screening\sounds\set8-2\'];
opt.cue_std = [sound_dir '5'];
% opt.cue_std = opt.cue_std*.25;
opt.cue_dev = {[sound_dir '1'] ...
    [sound_dir '9']};
% opt.cue_dev = opt.cue_dev*.25;
%testing
opt.bv_host = '';
opt.filename = '';
opt.test = 1;
target_list = [1 2];

for i = 1:length(target_list),
    stimutil_playMultiSound(wavread(opt.cue_dev{target_list(i)}), 'repeat', 3, 'pahandle', opt.pahandle, 'interval', 1, 'speakerCount', opt.speaker_number);
    opt.distractor = setdiff([1 2],target_list(i));
    pause(1);
    stim_oddballAuditory(N, opt);
    stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');    
end


%% RECORDING of 2-target oddball (ISI = 400)
N=300;
iterations = 1;
clear opt;
setup_auditory_screening;
opt.perc_dev = 20/100;
opt.isi = 400;
opt.use_speaker = [];
sound_dir = [BCI_DIR '\python\pyff\src\Feedbacks\Auditory_stimulus_screening\sounds\set8-2\'];
opt.cue_std = [sound_dir '5'];
% opt.cue_std = opt.cue_std*.25;
opt.cue_dev = {[sound_dir '1'] ...
    [sound_dir '9']};
% opt.cue_dev = opt.cue_dev*.25;
opt.filename = 'oddball2TargetMessung400';
opt.test = 0;
target_list = reshape([ones(1,iterations); ones(1,iterations)+1],1,[]);
target_list = target_list(randperm(length(target_list))); % if random is preferred. 
target_list = [1 2 target_list];
opt.speech_intro = [];
opt.impedances = 0;
opt.countdown = 0; %important !!

for i = 1:length(target_list),
    stimutil_playMultiSound(wavread(opt.cue_dev{target_list(i)}), 'repeat', 3, 'pahandle', opt.pahandle, 'interval', 1, 'speakerCount', opt.speaker_number);
    opt.distractor = setdiff([1 2],target_list(i));
    pause(1);
    stim_oddballAuditory(N, opt);
    stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');    
end

%% short example trial of 6 class oddball (ISI = 400)
N=30;
clear opt;
setup_auditory_screening;
opt.perc_dev = 1/6;
opt.isi = 400;
opt.use_speaker = [];
sound_dir = [BCI_DIR '\python\pyff\src\Feedbacks\Auditory_stimulus_screening\sounds\set8-2\'];
filenames = {[sound_dir '1'], ...
    [sound_dir '2'], ...
    [sound_dir '3'], ...
    [sound_dir '4'], ...
    [sound_dir '5'], ...
    [sound_dir '6']};
%testing
opt.bv_host = '';
opt.filename = '';
opt.test = 1;
target_list = [1 2];

for i = 1:length(target_list),
    opt.cue_dev = filenames{target_list(i)};
    opt.cue_std = {filenames{setdiff([1:length(filenames)],target_list(i))}};
    stimutil_playMultiSound(wavread(opt.cue_dev{1}), 'repeat', 3, 'pahandle', opt.pahandle, 'interval', 1, 'speakerCount', opt.speaker_number);
    pause(1);
    stim_oddballAuditory(N, opt);
    stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');    
end


%% RECORDING of 6 class oddball (ISI = 400)
N=300;
iterations = 1;
clear opt;
setup_auditory_screening;
opt.perc_dev = 1/6;
opt.isi = 400;
opt.use_speaker = [];
sound_dir = [BCI_DIR '\python\pyff\src\Feedbacks\Auditory_stimulus_screening\sounds\set8-2\'];
filenames = {[sound_dir '1'], ...
    [sound_dir '2'], ...
    [sound_dir '3'], ...
    [sound_dir '4'], ...
    [sound_dir '5'], ...
    [sound_dir '6']};
opt.filename = 'oddball6ClassMessung400';
opt.test = 0;
target_list = repmat([1:length(filenames)],1,iterations);
target_list = target_list(randperm(length(target_list))); % if random is preferred. 
% target_list = [1 2 target_list];
opt.speech_intro = [];
opt.impedances = 0;
opt.countdown = 0; %important !!

for i = 1:length(target_list),
    opt.cue_dev = filenames{target_list(i)};
    opt.cue_std = {filenames{setdiff([1:length(filenames)],target_list(i))}};
    stimutil_playMultiSound(wavread(opt.cue_dev{1}), 'repeat', 3, 'pahandle', opt.pahandle, 'interval', 1, 'speakerCount', opt.speaker_number);
    pause(1);
    stim_oddballAuditory(N, opt);
    stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');   
end

%% short example trial of 9 class oddball (ISI = 400)
N=30;
clear opt;
setup_auditory_screening;
opt.perc_dev = 1/9;
opt.isi = 400;
opt.use_speaker = [];
sound_dir = [BCI_DIR '\python\pyff\src\Feedbacks\Auditory_stimulus_screening\sounds\set8-2\'];
filenames = {[sound_dir '1'], ...
    [sound_dir '2'], ...
    [sound_dir '3'], ...
    [sound_dir '4'], ...
    [sound_dir '5'], ...
    [sound_dir '6'], ...
    [sound_dir '7'], ...
    [sound_dir '8'], ...
    [sound_dir '9']};
%testing
opt.bv_host = '';
opt.filename = '';
opt.test = 1;
target_list = [1 2];

for i = 1:length(target_list),
    opt.cue_dev = filenames{target_list(i)};
    opt.cue_std = {filenames{setdiff([1:length(filenames)],target_list(i))}};
    stimutil_playMultiSound(wavread(opt.cue_dev{1}), 'repeat', 3, 'pahandle', opt.pahandle, 'interval', 1, 'speakerCount', opt.speaker_number);
    pause(1);
    stim_oddballAuditory(N, opt);
    stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');    
end


%% RECORDING of 9 class oddball (ISI = 400)
N=300;
iterations = 1;
clear opt;
setup_auditory_screening;
opt.perc_dev = 1/9;
opt.isi = 400;
opt.use_speaker = [];
sound_dir = [BCI_DIR '\python\pyff\src\Feedbacks\Auditory_stimulus_screening\sounds\set8-2\'];
filenames = {[sound_dir '1'], ...
    [sound_dir '2'], ...
    [sound_dir '3'], ...
    [sound_dir '4'], ...
    [sound_dir '5'], ...
    [sound_dir '6'], ...
    [sound_dir '7'], ...
    [sound_dir '8'], ...
    [sound_dir '9']};
opt.filename = 'oddball9ClassMessung400';
opt.test = 0;
target_list = repmat([1:length(filenames)],1,iterations);
target_list = target_list(randperm(length(target_list))); % if random is preferred. 
% target_list = [1 2 target_list];
opt.speech_intro = [];
opt.impedances = 0;
opt.countdown = 0; %important !!

for i = 1:length(target_list),
    opt.cue_dev = filenames{target_list(i)};
    opt.cue_std = {filenames{setdiff([1:length(filenames)],target_list(i))}};
    stimutil_playMultiSound(wavread(opt.cue_dev{1}), 'repeat', 3, 'pahandle', opt.pahandle, 'interval', 1, 'speakerCount', opt.speaker_number);
    pause(1);
    stim_oddballAuditory(N, opt);
    stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');   
end
