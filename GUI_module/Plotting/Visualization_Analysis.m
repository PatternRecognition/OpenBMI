%% 사용자 입력
clear all; clc;
subject_name = {'sbsim', 'prchoi', 'eskim', 'smkang', 'bykim','yskim','ejlee','dblee'};
for i=1:length(subject_name)
%%
close all; 
filepath = 'C:\Users\cvpr\Documents\NEW_RAW_FILES';
subject_num = i;    %i
session_num = 1;

%% data import
subject = sprintf('subject%d_%s', subject_num, subject_name{subject_num}); 
session = sprintf('session%d', session_num);
subject_info.subject_num=subject_num;
subject_info.subject_name=subject_name;
subject_info.subject=subject;
subject_info.session_num=session_num;
subject_info.session=session;
% %지울것
% filename= 'p300_on';
%% plotting
p300_plotting(filepath,subject_info,'p300_off');
p300_plotting(filepath,subject_info, 'p300_on');
ERDERS(filepath,subject_info, 'mi_off');
ERDERS(filepath,subject_info, 'mi_on');
ssvep_fft_plot(filepath,subject_info, 'ssvep_off');
ssvep_fft_plot(filepath,subject_info,'ssvep_on');

end