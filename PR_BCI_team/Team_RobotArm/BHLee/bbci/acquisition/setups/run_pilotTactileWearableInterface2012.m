clear all
close all
clc 
startup

myname= mfilename;
session_name= myname(5:end);

fprintf('\n\nWelcome to the study "%s"!\n\n', session_name);
startup_new_bbci_online;
addpath([BCI_DIR 'acquisition/setups/' session_name]);

% USB2LPT
IO_ADDR= hex2dec('278');
new_subject=[];
VP_Code_present=[];

global TODAY_DIR; global VP_NUMBER

%% Start BrainVisionn Recorder, load workspace and check triggers
system('start Recorder'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', session_name);
try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

%% Create data folder/ VP-Code, nr etc.


while isempty(strmatch(new_subject, strvcat('Y', 'y', 'N', 'n'),'exact'))==1;
    new_subject=input('Is participant a new subject? Yes if this is a new experiment, no if the participant''s experiment was interrupted. [Y/N]', 's');
    if new_subject=='Y' || new_subject=='y'

        while isempty(strmatch(VP_Code_present, strvcat('Y', 'y', 'N', 'n'),'exact'))==1
            VP_Code_present=input('Does participant already have a VP-Code? [Y/N]', 's');
            if VP_Code_present== 'Y' %Falls VP schon Code hat:
                global VP_CODE; 
                VP_CODE= input('Please enter VP-Code, e.g. ''VPfae'': ');
            elseif VP_Code_present== 'N'
                warning('VP_CODE is undefined - assuming fresh subject');
            else
                fprintf('Please answer [Y/N].\n');
            end     
        end
        acq_makeDataFolder('multiple_folders', 1) 
        
   elseif new_subject=='N' || new_subject=='n'
       acq_makeDataFolder; %data will be saved in last folder created
   else
        fprintf('Please answer [Y/N].\n');
   end
end


VP_NUMBER= acq_vpcounter(session_name, 'new_vp');
RUN_END= [246 255]; %specific markers

% Display feedback on laptop screen
screen_pos= get(0, 'ScreenSize');
VP_SCREEN= [0 0 screen_pos(3:4)];

bvr_sendcommand('checkimpedances'); %AANZETTEN BIJ ECHTE RECORDING
stimutil_waitForInput('msg_name', 'when preparation of the cap is finished.');
bvr_sendcommand('viewsignals');

eval(['run_' session_name '_script']);

