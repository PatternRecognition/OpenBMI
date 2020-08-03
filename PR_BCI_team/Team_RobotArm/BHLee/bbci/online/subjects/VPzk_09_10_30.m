%   Variable 'bbci', struct with fields
%    train_file    a list of train_files
%    classDef      a cell array which defines the classes 
%                  {Tok1,Tok2,...; Name1, Name2,...}
%    setup         name of a setup (e.g. csp, selfpaced,...)
%                  (see directory setups)
%    player        the player number. Optional, default: 1
%    feedback      the name of a feedback
global ISCBF
ISCBF=1;
%bbci.train_file={'VPzk_09_10_30/training_ssvepVPzk','VPzk_09_10_30/training_ssvepVPzk02'};
bbci.train_file=('VPzk_09_10_30/training_ssvepVPzk*');
bbci.classDef={1,2,3,4,5,6,7,8;'4Hz','5Hz','6Hz','7.5Hz','8Hz','10Hz','12Hz','15Hz'};
bbci.setup='ssvep';
bbci.player=1;
bbci.feedback='ssvep';
bbci.fs=100;
bbci.save_name='VPzk_09_10_30/setup_ssvepVPzk';
