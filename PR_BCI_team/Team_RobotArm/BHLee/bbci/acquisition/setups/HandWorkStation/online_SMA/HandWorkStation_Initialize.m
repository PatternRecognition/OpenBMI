
function wld = HandWorkStation_Initialize(varargin)

wld.strategy = 'ranksum';

wld.bands = [1 3 ; 4 7 ; 8 15 ; 16 45];  % Hz
wld.tau = 70; % ms
wld.T_epo = 3000;   % ms
wld.block_offset = 30; % sec
wld.fs = 125;    % Hz
wld.fs_orig = 1000;
wld.filt_raw = [53 62];
wld.clab  = {'not','E*','R*','F9,10','FT7,8','T7,8','TP7,8','P9,10','Fp*','AF7,8'};

wld.wlen_sec = 30;  % sec
wld.force_ival_sec = 240; % sec

wld.mrk.start_low = 10;
wld.mrk.end_low = 11;
wld.mrk.start_high = 20;
wld.mrk.end_high = 21;
wld.mrk.force_down = 30;
wld.mrk.force_up = 31;
wld.mrk.control_down = 40;
wld.mrk.control_up = 41;

wld.speed.minmax = [3 10];
wld.speed.calibration = [4 9];
wld.speed.delta = 2;

wld.control_str= 'i:bbci_act_output';
wld.calibration.ISI = 2*60;
%wld.calibration.ISI_list = [120 100 100 240 120 160 240 200];
wld.calibration.ISI_list = repmat(120,1,12);
wld.calibration.nBlocks = 12;
wld.T = wld.calibration.ISI*wld.calibration.nBlocks;
