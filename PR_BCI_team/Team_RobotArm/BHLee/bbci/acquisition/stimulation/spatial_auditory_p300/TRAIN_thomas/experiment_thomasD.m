%% set everything up
addpath([BCI_DIR 'acquisition/stimulation/spatial_auditory_p300/TRAIN_thomas']);
% for testing only
%VP_CODE='VPtest';
%warning('using VP_CODE for testing');

setup_spatialbci_TRAIN_thomas
warning('turn the volume all the way up on the m-audio card and press enter to proceed');

input('');

% small screen for testing
VP_SCREEN=[-1920 0 1920 1200];
opt.position=VP_SCREEN;
%warning('using VP_SCREEN for testing');

saveName=[VP_CODE 'DurVariSlow'];
nSpeakers=6;

% start the recorder and load a workspace
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', ['reducerbox_64stdMastoid']);

opt.isi = 1000; % inter stimulus interval
opt.isi_jitter = 10; % defines jitter in ISI
opt.useSpeech=0;

round_lenght=10;
round_jitter=1;



% this prevents the auditoryMainroutine from starting or stoping the recorder
opt.recorderControl=0;

% the durations of the tones
durations = [ 5,  ...
              10,  ...
              20,  ...
              40, ...
              80, ...
              160, ...
              300];
            
% calibrated for loud_stages on speaker 1 
%loud_stages = [ 0.0035 0.04358 0.08643 0.1334 0.1899 0.2689 0.4];
% duration: 40 ms

volumes=[ 0.5 0.6 0.5 0.6 0.55 0.55 0 0;... % 5 ms
          0.23 0.25 0.22 0.24 0.23 0.25 0 0;... % 10
          0.15 0.15 0.15 0.15 0.14 0.155 0 0;... % 20
          0.1 0.115 0.1 0.12 0.115 0.115 0 0;... % 40
          0.08 0.085 0.085 0.085 0.085 0.085 0 0;... % 80
          0.07 0.08 0.07 0.085 0.07 0.07 0 0;... % 160
          0.06 0.07 0.07 0.08 0.07 0.07 0 0]; % 300    

% a target sequence is loaded

load(fullfile(BCI_DIR ,'acquisition\stimulation\spatial_auditory_p300\TRAIN_thomas\stimulus_sequence.mat'))



%% show the description
desc= stimutil_readDescription('instructions_experiment_thomasD');


stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to continue.');

%% demonstration

addpath([BCI_DIR 'acquisition/stimulation/spatial_auditory_p300/TRAIN_thomas']);
duration_index=floor(length(durations)*rand)+1;

opt.toneDuration=durations(duration_index);
opt=generateTones(opt);

opt.targetSeq=floor(rand*6)+1;
opt.isi=300;
opt.calibrated=diag(volumes(duration_index,:));
   % do the actual stimulation
opt.maxRounds=7;
opt.test=1;
   auditory_MainRoutine(opt);
   opt.test=0;
   opt.isi=1000;
   
%% do the first run

% struct to keep track of the responses
logVar=[];
impedance=1;
count_since_break=0;
blockCount=1;
for i=1:length(duration_sequence),
   addpath([BCI_DIR 'acquisition/stimulation/spatial_auditory_p300/TRAIN_thomas']);
   if count_since_break==0,
     % restart the recorder every 6 trials
     bvr_startrecording(saveName,'impedances',impedance);
     impedance=0;
   end
   % set the marker according to the duration
   opt.cueMarkerOffset=duration_sequence(i)*10;
   opt.targetMarkerOffset=duration_sequence(i)*10+100;
  
   opt.targetSeq=[target_sequence(i)];
   opt.maxRounds=round_lenght+round(rand)*round_jitter;
   
   
   disp(['trial ' num2str(i) ' of ' num2str(length(duration_sequence))]);
   disp(['the current duration is ', num2str(durations(duration_sequence(i)))]);
   disp(['the current loudness level is ', num2str(duration_sequence(i))]);
   disp(['the target count is ', num2str(opt.maxRounds)]);
   
   % generate the tones according to the duration
   opt.toneDuration=durations(duration_sequence(i));
   opt=generateTones(opt);
   
   % set the loudness for each speaker and duration
   opt.calibrated=diag(volumes(duration_sequence(i),:));
   % do the actual stimulation
   auditory_MainRoutine(opt);
   count_since_break=count_since_break+1;
   
   if count_since_break==6,
     count_since_break=0;
     bvr_sendcommand('stoprecording');
     disp('file written');
   end
   answer=input('enter the subjects answer\n');
   logVar(i).answer=answer;
   logVar(i).volume=duration_sequence(i);
   logVar(i).duration=durations(duration_sequence(i));
   logVar(i).count=opt.maxRounds;
   save(fullfile(TODAY_DIR,[saveName '_response']),'logVar');
   disp('type dbcont to continue');
   keyboard;
   
end

disp('end of first block');

%% do the second run


% a target sequence is loaded
load(fullfile(BCI_DIR ,'acquisition\stimulation\spatial_auditory_p300\TRAIN_thomas\stimulus_sequence.mat'))

logVar=[];
input('press enter to start the second block');
saveName=[VP_CODE 'DurVariFast'];

impedance=1;
opt.isi = 300; % inter stimulus interval
opt.isi_jitter=10;
count_since_break=0;
blockCount=1;
for i=1:length(duration_sequence),
   addpath([BCI_DIR 'acquisition/stimulation/spatial_auditory_p300/TRAIN_thomas']);
   if count_since_break==0,
     % restart the recorder every 12 trials
     bvr_startrecording(saveName,'impedances',impedance);
     impedance=0;
   end
   % set the marker according to the duration
   opt.cueMarkerOffset=duration_sequence(i)*10;
   opt.targetMarkerOffset=duration_sequence(i)*10+100;
  
   opt.targetSeq=[target_sequence(i)];
   opt.maxRounds=round_lenght+round(rand)*round_jitter;
   
   
   disp(['trial ' num2str(i) ' of ' num2str(length(duration_sequence))]);
   disp(['the current duration is ', num2str(durations(duration_sequence(i)))]);
   disp(['the current loudness level is ', num2str(duration_sequence(i))]);
   disp(['the target count is ', num2str(opt.maxRounds)]);
   
   % generate the tones according to the duration
   opt.toneDuration=durations(duration_sequence(i));
   opt=generateTones(opt);
   
   % set the loudness for each speaker and duration
   opt.calibrated=diag(volumes(duration_sequence(i),:));
   % do the actual stimulation
   auditory_MainRoutine(opt);
   count_since_break=count_since_break+1;
   
   if count_since_break==12,
     count_since_break=0;
     bvr_sendcommand('stoprecording');
     disp('file written');
     
   end
   answer=input('enter the subjects answer \n');
   logVar(i).answer=answer;
   logVar(i).volume=duration_sequence(i);
   logVar(i).duration=durations(duration_sequence(i));
   logVar(i).count=opt.maxRounds;
   save(fullfile(TODAY_DIR,[saveName '_response']),'logVar');
   disp('type dbcont to continue');
   keyboard;
  
end

disp('end of second block');


 

