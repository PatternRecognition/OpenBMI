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

%% vis ERP 3 km/s
stimulusSTART=[padding; uint64(102); timestamp];    % start trigger
stimulusEND=[padding; uint64(202); timestamp];    % end trigger

write(tcp_ear, stimulusSTART);
ppWrite(IO_ADD,102); 
% 
vis_ERP_paradigm_ambulatory(trig_vis_erp, nSequence)
%   
write(tcp_ear, stimulusEND);
ppWrite(IO_ADD,202);

%% vis ERP 6 km/h
stimulusSTART=[padding; uint64(103); timestamp];    % start trigger
stimulusEND=[padding; uint64(203); timestamp];    % end trigger

write(tcp_ear, stimulusSTART);  
ppWrite(IO_ADD,103);
%    
vis_ERP_paradigm_ambulatory(trig_vis_erp, nSequence)
%
write(tcp_ear, stimulusEND);
ppWrite(IO_ADD,203);
%% %%%%%%%%%%%%%%%%%%%%  Auditory ERP   %%%%%%%%%%%%%%%%%%%%%%%%%%
%% setting
% class = {'stand','1.6', '2.0'}; % 0 3 6 km/h 
% trigger num
trig_erp = [1 2]; % standing condition
% trig_walk_08 = [21 22]; % walking condition 0.8 m/s
% trig_walk_16 = [31 32]; % walking condition 1.6 m/s

% 300
nTrial = 300;

%  example_aud_ERP(300)   
%% aud ERP 0 km/s
stimulusSTART=[padding; uint64(301); timestamp];    % start trigger
stimulusEND=[padding; uint64(401); timestamp];     % end trigger
        
write(tcp_ear, stimulusSTART);
ppWrite(IO_ADD,301); 
%
aud_ERP_paradigm_ambulatory(trig_erp, nTrial)
%  
write(tcp_ear, stimulusEND);
ppWrite(IO_ADD,401);

%% aud ERP 3 km/s
stimulusSTART=[padding; uint64(302); timestamp];    % start trigger
stimulusEND=[padding; uint64(402) ; timestamp];    % end trigger
      
write(tcp_ear, stimulusSTART);
ppWrite(IO_ADD,302);  
% 
aud_ERP_paradigm_ambulatory(trig_erp, nTrial)
%   
write(tcp_ear, stimulusEND);
ppWrite(IO_ADD,402);

%% aud ERP 6 km/h   
stimulusSTART=[padding; uint64(303); timestamp];    % start trigger
stimulusEND=[padding; uint64(403); timestamp];    % end trigger

write(tcp_ear, stimulusSTART);  
ppWrite(IO_ADD,303);
%    
aud_ERP_paradigm_ambulatory(trig_erp, nTrial)
%
write(tcp_ear, stimulusEND);
ppWrite(IO_ADD,403);
%% %%%%%%%%%%%%%%%%%%%%%%   SSVEP   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


