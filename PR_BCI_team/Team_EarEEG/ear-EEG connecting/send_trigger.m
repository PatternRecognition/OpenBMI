clear all; clc;

%% 프로그램 세팅
%   openvibe acquisition sever
%       Driver: mBrainTrain Smarting
%       Driver Properties
%           number of channels: 24
%           port number: 5
%           sampling frequency: 500
%           change channel name -> load -> 내문서\ear_chan_name.csv
%
%       Preferences
%           Select only named channels: check
%           enable External Stimulation: check
%           LSL_EnableSLOutput: check
%
%       connect -> play
%       => 연결 완료
%
%   openvibe designer
%       열기- 파일명:design2_TCP.mxs
%       재생
%       => 저장, display, matlab 연결
%
%% matlab setting
%   경로추가
%       matlab-openvibe 통신
%           external\liblsl-Matlab
%       openvibe에서 online data 받아오기
%           external\eeglab_10_0_1_0x
%       EEG plot하기, brain vision에서 online data 가져오기
%           C:\Users\cvpr\Documents\OpenBMI-master

%% send trigger
% brain vision setting
global IO_LIB IO_ADD;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec('E010');

% openviber setting
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
