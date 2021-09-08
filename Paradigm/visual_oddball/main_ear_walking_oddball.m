clear all; close all; clc

%%
global IO_LIB IO_ADD;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec('D010');

% openvibe trigger setting
tcp_ear = tcpclient('localhost', 15361);
padding=uint64(0);
timestamp=uint64(0);

%% %%%%%%%%%%%%%%%%%%%%  visual ERP   %%%%%%%%%%%%%%%%%%%%%%%%%%
%% setting
% trigger num
trig_vis_erp = [1 2]; % standing condition
nSequence = 6;

% example_vis_ERP(6);
%% vis ERP 0 km/s
stimulusSTART=[padding; uint64(101); timestamp];    % start trigger
stimulusEND=[padding; uint64(201); timestamp];     % end trigger
     
write(tcp_ear, stimulusSTART);
ppWrite(IO_ADD,101);  
%
vis_ERP_paradigm_ambulatory(trig_vis_erp, nSequence)
%
write(tcp_ear, stimulusEND);
ppWrite(IO_ADD,201);


