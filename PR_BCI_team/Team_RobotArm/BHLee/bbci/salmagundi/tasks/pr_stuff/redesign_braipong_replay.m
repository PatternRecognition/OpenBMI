cd('D:\EEG_Video\brainpong_replays');
addpath([BCI_DIR 'bbci_alf/log']);

global LOG_DIR
LOG_DIR= [EEG_RAW_DIR 'Guido_05_04_01/log/'];

logno= 17;

clear opt
%opt.position= [2 552 640 480];
%opt.position= [2 -1010 800 600];
opt.position= [2 580 800 600];
opt.blackfield= 1;
opt.freeze_end= 5;
opt.max_length= 90;
if ~isunix,
  opt.bat1_sound= [DATA_DIR 'audio/G5.wav'];
  opt.bat2_sound= [DATA_DIR 'audio/G5.wav'];
  opt.border_sound= [DATA_DIR 'audio/E5.wav'];
  opt.ballout_sound= [DATA_DIR 'audio/A5.wav'];
end


%logno= 10; start= 34.8; stop= 109;
%replay_brainpong_new(logno, opt, 'start',start, 'stop',stop);

%logno= 13; start= 47.3; stop= 118;
%replay_brainpong_new(logno, opt, 'start',start, 'stop',stop);

%%guter start, dann schlecht
%logno= 9; start= 192.4; stop= 261;
%replay_brainpong_new(logno, opt, 'start',start, 'stop',stop);


logno= 8; start= 53.5; stop= 448;
replay_brainpong_new(logno, opt, 'start',start, 'stop',stop, 'batheight',0.11, ...
		     'save',['bpong' int2str(logno)]);

logno= 15; start= 35.3; stop= 289;
replay_brainpong_new(logno, opt, 'start',start, 'stop',stop, ...
		     'save',['bpong' int2str(logno)]);

logno= 16; start= 31.9; stop= 241;
replay_brainpong_new(logno, opt, 'start',start, 'stop',stop, ...
		     'save',['bpong' int2str(logno)]);

logno= 17; start= 17.9; stop= 264;
replay_brainpong_new(logno, opt, 'start',start, 'stop',stop, ...
		     'save',['bpong' int2str(logno)]);

logno= 17; start= 17.9;
replay_brainpong_new(logno, opt, 'start',start, 'stop',start+30, ...
		     'save','bpong_demo', 'freeze_end',0);
   