
function wld = HandWorkStationGame_Initialize(varargin)

wld.strategy = 'ranksum';

wld.bands = [1 3 ; 4 7 ; 8 15 ; 16 45];  % Hz
wld.T_epo = 3000;   % ms
wld.fs = 125;    % Hz
wld.fs_orig = 1000;
wld.filt_raw = [53 62];
wld.clab  = {'not','E*','R*'};

wld.wlen_sec = 15;  % sec
wld.force_ival_sec = 40; % sec

wld.mrk.start_low = 10;
wld.mrk.end_low = 11;
wld.mrk.start_high = 20;
wld.mrk.end_high = 21;
wld.mrk.force_down = 30;
wld.mrk.force_up = 31;
wld.mrk.control_down = 40;
wld.mrk.control_up = 41;

wld.speed.minmax = [1 7];
wld.speed.calibration = [2 6];
wld.speed.delta = 1;

wld.calibration.ISI = 30;
wld.calibration.nBlocks = 4;
wld.T = wld.calibration.ISI*wld.calibration.nBlocks;

