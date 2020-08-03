%% set everything up

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end
setup_spatialbci_TRAIN_thomas
% start the recorder and load a workspace
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', ['reducerbox_64mcc_noEOG']);

pause(3); 

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

% set up the data directories
global TODAY_DIR REMOTE_RAW_DIR
acq_makeDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;
LOG_DIR = [TODAY_DIR '\log\'];


%VP_SCREEN= [-1281 0 1280 1024];
VP_SCREEN=[0 0 500 400];

fileName=['loudness_',VP_CODE];

% initialize the timer
waitForSync;


%% configure the experiment

minLoud=1e-5;
maxLoud=0.7;
numLoud=15;

spacing='log';

isi=1500;
isiJitter=500;


% create a sequence of loudness levels
if strcmp(spacing,'linear'),
  loudness=linspace(minLoud,maxLoud,numLoud);
elseif strcmp(spacing,'log'),
  loudness=logspace(log10(minLoud),log10(maxLoud),numLoud).^0.5;
end
loudSeq=zeros(6*length(loudness),8,8);
count=1;
for i=1:length(loudness),
  for speaker=1:6
    tone=zeros(1,8);
    tone(speaker)=loudness(i);
    loudSeq(count,:,:)=diag(tone);
    count=count+1;
  end
end
loudSeq=loudSeq(randperm(size(loudSeq,1)),:,:);

%% start the experiment

input('press enter to start the measurement')

% start the recorder
actualFileName=bvr_startrecording(fileName);
pause(10)
% the start marker
ppTrigger(10);
disp('starting loudness test ...');
% the sound vectors
waves=[opt.cueStream; zeros(2,size(opt.cueStream,2))];
for i=1:size(loudSeq,1)
  % extract loudness and speaker number from sequence
  tone = squeeze(loudSeq(i,:,:));
  speaker=find(sum(tone,1));
  % play the sound
  PsychPortAudio('FillBuffer', opt.pahandle, (waves'*tone)');
  PsychPortAudio('Start', opt.pahandle);
  PsychPortAudio('Stop', opt.pahandle, 1);
  % send a stimulus marker to the eeg
  ppTrigger(10+speaker);
  % wait the appropriate time
  time=isi+(2*rand-1)*isiJitter;
  
  waitForSync(time)
  
  
    
end
% wait a few seconds in case there is another response
pause(4);
% send an end marker
ppTrigger(17);

% stop the recording
bvr_sendcommand('stoprecording');

% save the sequence of loudnesses to a file

savename=[actualFileName,'_values'];
save(savename,'loudSeq');
%% adaptation
% calculate preliminary thresholds and create a new target sequence
% cluster new loudness values around that estimate
disp('calculating preliminary threshold');
threshold=process_loudness(actualFileName);

num_levels=10;
max_factor=2;

loudSeq=zeros(num_levels*6,8,8);
index=1;
for i=1:length(threshold),
  
  loud=linspace(minLoud,max_factor*threshold(i),num_levels);
  
  for j=1:num_levels,
    loudSeq(index,i,i)=loud(j);
    index=index+1;
  end  
end
loudSeq=loudSeq(randperm(size(loudSeq,1)),:,:);
%% second block
disp('staring second block')
actualFileName=bvr_startrecording(fileName,'impedances',0);
pause(1)
% the start marker
ppTrigger(10);
disp('starting loudness test ...');
% the sound vectors
waves=[opt.cueStream; zeros(2,size(opt.cueStream,2))];
for i=1:size(loudSeq,1)
  % extract loudness and speaker number from sequence
  tone = squeeze(loudSeq(i,:,:));
  speaker=find(sum(tone,1));
  % play the sound
  PsychPortAudio('FillBuffer', opt.pahandle, (waves'*tone)');
  PsychPortAudio('Start', opt.pahandle);
  PsychPortAudio('Stop', opt.pahandle, 1);
  % send a stimulus marker to the eeg
  ppTrigger(10+speaker);
  % wait the appropriate time
  time=isi+(2*rand-1)*isiJitter;
  
  waitForSync(time)
  
  
    
end
% wait a few seconds in case there is another response
pause(4);
% send an end marker
ppTrigger(17);

% stop the recording
bvr_sendcommand('stoprecording');

% save the sequence of loudnesses to a file

savename=[actualFileName,'_values'];
save(savename,'loudSeq');

%% get the actual result and save thresholds to file
threshold=process_loudness(actualFileName,1);
savename=[actualFileName,'_threshold'];
save(savename,'threshold');
%%
disp(' ');
disp('loudness test completed');
disp('thank you');
disp(' ');
disp(['results have been saved to ' savename]);








