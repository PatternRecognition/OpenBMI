%% %%%%%%%%%%%%%%%%%%%%%%   SSVEP   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 12, 8.57, 5.45, 3.53 Hz 
trig_ssvep = [1 2 3]; % standing condition
% 20 or 50
nTrial = 20; % 3 class * 15 = 45 trials

% example_SSVEP(20);
% SSVEP TR TE
 
% trigger, num of tr, comm, screen num (3), session num (shuffle)
SSVEP_paradigm_ambulatory(trig_ssvep,nTrial,[0 0 0],0, 1);



