% Auditory ISI-experiment run-script
% run_TOBI_students_isi

%% Initialization
%TODO everything... call play_auditory_oddball_ISI with the specific
%parameters

opt = [];
opt.toneDuration = 40;
opt.speakerSelected = [6 2 4 1 5 3];
opt.language = 'german';

%setup_spatialbci_GLOBAL

opt.isi_jitter = 0; % defines jitter in ISI

opt.itType = 'fixed';
opt.mode = 'copy';
opt.application = 'TRAIN';

opt.countdown = 0;
% opt.repeatTarget = 3;

opt.require_response = 0;

opt.fixation = 1;
opt.filename = 'auditory_isi';
opt.speech_intro = '';
opt.fixation = 1;
opt.fs = 44100;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = opt.cue_std*.25;
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = opt.cue_dev*.25;


nbblocks = 5;
isivec = [1000 400 275 225 175 125 75 50]; % isi given in ms (index 1: larger, index 8: smaller)
isivec = [1000 400 275 225 200 175 150 125 112 100 87 75 62 50]; % isi given in ms (index 1: larger, index 8: smaller)
fprintf('Finished initialization. \n')


%% while preparing the cap, we show the stimuli to the participant
disp('while preparing the cap, we show the stimuli to the participant')
pause;
preBlocks = [1000 275 50];
opt.recording = 0;
for myisi = preBlocks
    opt.filename = [];
    opt.isi = myisi;
    seq = accseq;
    play_auditory_oddball_ISI(seq, opt);
    sprintf('how many did you count (TRUE NUMBER: %i, ISI %i)?\n',sum(seq), opt.isi)
    pause
end

disp('finished preBlocks.')

%% standard measurement
%bvr_checkparport;


disp('start standard measurement')
disp('EYES OPEN - check for fixation cross, press <ENTER> to start')
pause;

bvr_startrecording(['eyesOpen_' VP_CODE], 'impedances', 1);
pause(90);
bvr_sendcommand('stoprecording');


sprintf('\n \n \n \n EYES CLOSED, press <ENTER> to start')
pause;
bvr_startrecording(['eyesClosed_' VP_CODE], 'impedances', 0);
pause(90);
bvr_sendcommand('stoprecording');

%% perform actual experiment with recording !!
% blocks = gblock(nbblocks);
nRep = 4;
blocks = [];
for ii = 1:nbblocks
    tmp = repmat(randperm(length(isivec)), [nRep, 1]); %matrix with nRep equal rows
    blocks(ii,:) = tmp(:); % parse the matrix column-wise sothat all nRep repititions are next to each other
end

for i = 1:size(blocks,1)
    for j = 1:size(blocks,2)
%         l = 0;
        opt.isi = isivec(blocks(i,j));
        opt.filename = ['auditory_isi_' num2str(opt.isi) '_'];
        opt.impedances = 0;
        seq = accseq;
        if opt.isi == 400
            seq = seq(1:floor(0.7 * length(seq)) );
        end
        if opt.isi == 1000
            seq = seq(1:floor(0.4 * length(seq)) );
        end
        
        sprintf('press <RETURN> to proceed with the next trial \n NEXT ISI: %d', opt.isi)        
        pause;
        play_auditory_oddball_ISI(seq, opt);
        sprintf('how many did you count (TRUE NUMBER: %i, ISI %i)?\n',sum(seq), opt.isi)
        
    end
%     input('Special Recording for ISI = 50ms \nPress ENTER to continue to the next block')
%     for k = 1:4, % 4x nebeneinander f�r special recording
%         opt.isi = 50;
%         opt.filename = ['auditory_isi_SPECIAL_' num2str(opt.isi) '_'];
%         opt.impedances = 0;
%         seq = accseq(12,125,20); % oder was anders
%         sprintf('press <RETURN> to proceed with the next trial \n NEXT ISI: %d', opt.isi)        
%         pause;
%         play_auditory_oddball_ISI(seq, opt);
%         sprintf('how many did you count (TRUE NUMBER: %i, ISI %i)?\n',sum(seq), opt.isi)
%     end
    input('Take a pause (5-10 min) \nPress ENTER to continue to the next block')
end
