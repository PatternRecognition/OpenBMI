
% %% Duration of CenterSpeller:
% nErrors = 100;
% ErrorRate = 0.25;
% nSequences = 4;
% nCountdown = 3;
% tAnimation = 0.5;
% tStimulus = 0.215;
% tInterstimulus = 0;
% 
% nCharacters = ceil(nErrors/ErrorRate/2);
% nTrials = nCharacters*2;
% tEndlevel1 = 2 + 2*tAnimation;
% tEndlevel2 = 2 + tAnimation;
% tCountdown = nCountdown + 2*tAnimation;
% tSequence = 6*(tStimulus + tInterstimulus);
% 
% %             countdown    n stimuli per level    level switch   feedback
% tCharacter = tCountdown + 2*nSequences*tSequence + tEndlevel1 + tEndlevel2;
% 
% tExperiment = nCharacters*tCharacter;
% tBlocks = 20*60;
% nBlocks = ceil(tExperiment/tBlocks);
% tBreak = 5*60;
% tExperimentWithBreaks = nBlocks*  tBreak;
% 
% 
% h=floor(tExperiment/3600);
% m=floor((tExperiment-h*3600)/60);
% s=tExperiment-h*3600-m*60;
% fprintf('\n==========================================================================\nnTrials: %d, nErrors: %.1f\n', nTrials, nErrors);
% fprintf('duration of CenterSpeller experiment:\t\t\t\t%d:%d:%.2f\n', h,m,s);
% 
% h=floor((tExperiment+tExperimentWithBreaks)/3600);
% m=floor(((tExperiment+tExperimentWithBreaks)-h*3600)/60);
% s=(tExperiment+tExperimentWithBreaks)-h*3600-m*60;
% fprintf('In %d blocks of %d mins with %d mins break, the duration is:\t%d:%d:%.2f\n', nBlocks, tBlocks/60, tBreak/60,h,m,s)


%% Duration of ErrP-calibration with moving arrow:

nCharacters = 500; %ceil(nErrors/ErrorRate/2);
ErrorRate = 0.2;
nErrors = int32(nCharacters*2 * ErrorRate); % 200;
nTrials = nCharacters*2;

% additional trials due to errors:
frac_AdditionalTrialsPerChar = ErrorRate * (2 + ...        % wrong group in level 1
                                           (29/36)*4 + ... % wrong letter in level 2
                                           (1/36)*2 + ...  % < by mistake
                                           (6/36)*2);      % backdoor by mistake
nTrials = nCharacters*2 + ...
          nCharacters*frac_AdditionalTrialsPerChar + ...
          ((nCharacters*frac_AdditionalTrialsPerChar)-nCharacters) * frac_AdditionalTrialsPerChar;

tArrowMoving = 1.47;
tFeedback_l1 = 2.36;
tFeedback_l2 = 2.35;

tCharacter = 2*tArrowMoving + tFeedback_l1 + tFeedback_l2;
tTrial = tArrowMoving + (tFeedback_l1 + tFeedback_l2)/2;

% tExperiment = nCharacters*tCharacter;
tExperiment = nTrials * tTrial;
% 
% tBlocks = 20*60;
% nBlocks = ceil(tExperiment/tBlocks);
% tBreak = 5*60;
% tExperimentWithBreaks = nBlocks*  tBreak;

h=floor(tExperiment/3600);
m=floor((tExperiment-h*3600)/60);
s=tExperiment-h*3600-m*60;
fprintf('\nnTrials: %d, nErrors: %.1f\n', int32(nTrials), int32(nErrors));
fprintf('duration of experiment:\n')
% fprintf('%d trials a %.2f seconds\n', int32(nTrials), tTrial);
fprintf('%d trials a %.2f seconds => \t\t\t\t%d:%d:%.2f\n', int32(nTrials), tTrial, h,m,s);

% h=floor((tExperiment+tExperimentWithBreaks)/3600);
% m=floor(((tExperiment+tExperimentWithBreaks)-h*3600)/60);
% s=(tExperiment+tExperimentWithBreaks)-h*3600-m*60;
% fprintf('In %d blocks of %d mins with %d mins break, the duration is:\t%d:%d:%.2f\n\n', nBlocks, tBlocks/60, tBreak/60,h,m,s)








