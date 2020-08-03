%% set everything up

VP_CODE='VPtest';
warning('using vp_code for testing');
setup_spatialbci_TRAIN_thomas
VP_SCREEN=[0 0 500 400];
warning('using vp_screen for testing');

language=1;   % language of the feedback (1 ger, 2 eng)
minLevel=0;   % minimum sound level
maxLevel=1;   % maximum sound level

graphics_cue=1;
audio_cue=1;

count_in=3;                % number of cues before tone presentation
isi=700;                   % time between cue stimuli
cue_blip=300;              % time for which the graphical cue is displayed
stim_blip=1000;            % time for which the graphical stimulus is displayed
time_before_trial=2000;    % wait this long between trials
time_before_question=2000; % wait this long after trial before asking question
num_equal=10;      % number of equally spaced loudness levels
num_per_level=2;   % number of presentations for each direction and loudness level
num_clustered=10;  % number of loudness levels clustered around upper and lower levels

audio_cue= diag([0.4 0.4 0.4 0.4 0.4 0.4  0 0]); % loudnes of the cue stimuli

% generate a loudness sequence
loudness=repmat(minLevel:(maxLevel-minLevel)/(num_equal-1):maxLevel,1,num_per_level);

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




question = {'Haben Sie einen Ton gehört?';...
  'Did you hear a sound?'};
answers  = {'Nein','Ja','Ja, und es war unangenehm laut';
  'No' ,'Yes','Yes, and it was uncomfortably loud'};

stim=[];
answer=[];
%% setup graphics
close all
% create the figure
g.fig=figure('position',VP_SCREEN,...
  'ToolBar','none',...
  'MenuBar','none',...
  'Color',[0 0 0]);
g.ax=axes('position',[0 0 1 1],...
  'XLim',[-1,1],...
  'YLim',[-1,1],...
  'Color',[0 0 0]);
axis('square');

% draw a circular patch as cue
num=40;
col=[ 1 0 0];
rad=0.2;
angle=0:pi/20:2*pi;
x=rad*cos(angle);
y=rad*sin(angle);

g.cue=patch(x,y,col);
set(g.cue,'Visible','off');





%% start the experiment
input('get the cap ready and press ENTER to start the experiment.\n')
disp('starting...')

% this controls what happens next
state=0;
%0: waiting for next trial
%1: play the cue
%2: play the stimulus
%3: ask the question

% extract the audio signals from the opt structure
waves=[opt.cueStream; zeros(2,size(opt.cueStream,2))];
count=1;
cues_presented=0;
waitForSync;
while 1,
  if state==0,% wait until trial starts
    waitForSync(time_before_trial);
    state=1;
  elseif state==1,% present a cue

    if cues_presented<count_in,
      cues_presented=cues_presented+1;
      PsychPortAudio('FillBuffer', opt.pahandle, (waves'*audio_cue)');
      PsychPortAudio('Start', opt.pahandle);

      PsychPortAudio('Stop', opt.pahandle, 1);
      if graphics,
        set(g.cue,'Visible','on');
        drawnow;
        waitForSync(cue_blip);
        set(g.cue,'Visible','off');
        drawnow;
        waitForSync(isi-cue_blip);
      else,
        waitForSync(isi);
      end
    else,
      cues_presented=0;
      state=2
    end
  elseif state==2,
    % extract the next stimulus from the sequence 
    tone = squeeze(loudSeq(count,:,:));
    speaker=find(sum(tone,1));
    % remember the loudness and direction
    stim=[stim;speaker,max(tone)];
    count=count+1;
  end
end




