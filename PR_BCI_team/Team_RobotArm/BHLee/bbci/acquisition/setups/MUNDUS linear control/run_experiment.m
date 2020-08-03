% INIT

global general_port_fields

LEFT_VS_FOOT = 1;
FOOT_VS_RIGHT = 2;

fb = struct();

general_port_fields = struct(   'bvmachine','127.0.0.1',...
                                'control',{{'127.0.0.1',12471,12487}},...
                                'graphic',{{'',12487}});
                        
general_port_fields.feedback_receiver= 'pyff';
general_port_fields.bvmachine = '127.0.0.1';           
    
% Get VP_CODE and create recording directory
global TODAY_DIR;
VP_CODE = 'VPtmp';
acq_makeDataFolder();

% Start PYFF
%FEEDBACKS_DIR = 'D:\svn\bbci\python\pyff\src\Feedbacks';
%PYFF_DIR = 'D:\svn\pyff\src';
%CFG_DIR = 'D:\svn\bbci\python\pyff\src\Feedbacks\BGUI\'

FEEDBACKS_DIR = 'D:\development\2011.5\src\Feedbacks'
PYFF_DIR = 'D:\development\mundus\runtime\pyff\src'
CFG_DIR = 'D:\development\2011.5\src\Feedbacks\BGUI\'

pyff('startup','dir',PYFF_DIR,'a',FEEDBACKS_DIR, 'bvplugin',0 );

%% CALIBRATION
pyff('init','FeedbackCursorArrowFES');
pyff('load_settings', 'D:\svn\bbci\python\pyff\src\Feedbacks\BGUI\feedback_base');
pyff('play');%, 'basename', 'calibMI', 'impedances', 0)

%% CLASSIFIER TRAINING

% choose classes (0 to test all combinations)
cl = 0;

if(cl == LEFT_VS_FOOT),
    channels = 'FC1;FCz;FC2;CFC1;CFC2;C1;Cz;C2;CCP1;CCP2;CP1;CPz;CP2;FC6;FC4;C6;C4;CP6;CP4;CFC8;CFC6;CFC4;CCP8;CCP6;CCP4';
    comb = 'left;foot';
    fb.fbclasses =  {'left', 'foot'};
elseif(cl == FOOT_VS_RIGHT),
    channels = 'FC1;FCz;FC2;CFC1;CFC2;C1;Cz;C2;CCP1;CCP2;CP1;CPz;CP2;FC5;FC3;C5;C3;CP5;CP3;CFC7;CFC5;CFC3;CCP7;CCP5;CCP3';
    comb = 'foot;right';
    fb.fbclasses =  {'foot', 'right'};
else,
    channels = '';
    comb = 'auto';
    fb.fbclasses =  {'left', 'right', 'foot'};
end;
        
[bbci, data] = bbci_calibrate_MI(TODAY_DIR, 'calibMI*', 'cfy_best_comb', channels,'','',comb)
bbci_save(bbci, data);

%% ONLINE

fb.log_filename = [TODAY_DIR '\bgui.log']

pyff('startup','dir',PYFF_DIR,'a',FEEDBACKS_DIR, 'bvplugin',0 );

pyff('init','BGUI');
pyff('load_settings', [CFG_DIR 'feedback_base']);
pyff('load_settings', [CFG_DIR 'feedback_online_LinCtrl']);
pyff('set',fb);
pyff('play')%, 'basename', 'onlineMI', 'impedances', 0)

bbci_acquire_bv('close')
bbci_online_multi(TODAY_DIR, 'cfy_best_comb', '0');