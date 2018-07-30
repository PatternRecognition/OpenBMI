clear all; clc;

%% Openvibe setting
%   openvibe acquisition sever
%       * Driver: mBrainTrain Smarting
%       * Driver Properties
%           number of channels: 27 (24 for ear-EEG signals and 3 for gyro)
%           port number: 5
%           sampling frequency: 500
%           change channel name -> load -> Document\ear_chan_name.csv
%
%       * Preferences
%           Select only named channels: check
%           enable External Stimulation: check
%           LSL_EnableSLOutput: check
%
%       * connect -> play
%       => connection complete!
%
%   openvibe designer
%       * open - filename: design2_TCP.mxs
%       * play
%       => save, display, matlab connection
%
%% matlab setting (external files)
%   add path in matlab for communication 
%       matlab-openvibe communication
%           > external\liblsl-Matlab
%       get online data from openvibe
%           > external\eeglab_10_0_1_0x

%% An example of sending trigger
% trigger setting for brain vision 
global IO_LIB IO_ADD;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec('E010');

% trigger setting for openvibe
t = tcpclient('localhost', 15361);
padding=uint64(0);
timestamp=uint64(0);
stimulus1=[padding; uint64(1); timestamp];
stimulus2=[padding; uint64(2); timestamp];
stimulus3=[padding; uint64(3); timestamp];

% send trigger
while true
    write(t, stimulus1);
    ppWrite(IO_ADD, 1);
    pause(1);
    
    write(t, stimulus2);
    ppWrite(IO_ADD, 2);
    pause(1);
    
    write(t, stimulus3);
    ppWrite(IO_ADD, 3);
    pause(1);
end
