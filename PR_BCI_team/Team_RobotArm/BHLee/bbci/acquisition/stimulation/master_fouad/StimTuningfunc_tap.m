% Para_ref_time:        reference intervallet i hver trial              [scalar]
% Para.act_time:        Length of each frequency representaion in sec   [scalar]
% Para.fs:              Sampling-frequency in Hz                        [scalar]
% Para.modfreq:         modulation freq in Hz                           [vector] 
% Para.carfreq:         carrier freq in Hz                              [scalar]  
% Para.num_trial:       number of trials                                [scalar]
% Para.count_dura:      number of countdown sec.                        [scalar]
% Para.num_block
%
% TRIGGERS:
% 000   : Begin session
% 001   : Begin each block
% 002   : Begin trial/ begin ref interval
% 003   : End ref interval
% 004   : begin 1 stim / begin pause interval between stim
% 0041  : End pause interval/ begin next stim interval
% 004   : End 1 stim/ begin pause interval between stim
% 0042  : End pause interval/ begin next stim interval
% .
% .
% 005   : End Trial 
% 006   : End Block      

% 007   : End Session
% Last modified 7/4-2008
% Fouad Channir, DTU/TUB, SSSEP BCI, Master Project

function StimTuningfunc_tap(Para,opt)

if isfield(Para, 'filename') && ~isempty(Para.filename),
  global VP_CODE
  bvr_startrecording([Para.filename VP_CODE]);
  else
    warning('!*NOT* recording: opt.filename is empty');
end

len = length(Para.modfreq);
[H_AX, RESTORE_SPEC]= stimutil_initFigure('position',[-1919 5 1920 1210]);
stimutil_countdown(Para.count_dura,opt)
drawnow;
for iii = 1:Para.num_block,
ppTrigger(101)
% opt.handle_background= stimutil_initFigure(opt);

h_msg= stimutil_fixationCross(opt);
set(h_msg, 'Visible','on');
drawnow;
clf
waitForSync;
for ii =1:Para.num_trial,
       indices = randperm(len);     
    mod_freq = Para.modfreq(indices);   % The different modulation frequencies...                                     % ... are arranged in random order

ppTrigger(102)
    waitForSync(Para.ref_time*1000)    % Length of reference period in msec.

    for i = 1:len,
    

ppTrigger(mod_freq(i));


        signal(:,1)=wavread(['D:\svn\bbci\acquisition\data\sound\cue_sssep\cue_tuning_',int2str(mod_freq(i))]);
        signal(:,2)=wavread(['D:\svn\bbci\acquisition\data\sound\cue_sssep\cue_tuning_',int2str(mod_freq(i))]);
        
        wavplay(signal,Para.fs,'async')  ;      

        waitForSync(Para.act_time*1000); 
ppTrigger(103)  
        waitForSync(Para.ifi*1000);
    end

end
ppTrigger(104)

if iii < Para.num_block
%     disp('Press any key to begin next block')
  drawnow;

stimutil_break(opt)
    
else 
    disp('DONE')
end

end
bvr_sendcommand('stoprecording');
% 
% pause(5);
% delete(h_msg);
