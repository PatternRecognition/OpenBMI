
%%% Parameters from Calibration_1 and Calibration_2

SPEED  = .99;
DIST_H = .3;
DIST_L = .425;


%%% start Pyff
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);


%%% 4 BIG BLOCKS
for ii = 1:1
    
    pyff('init','HandWorkStation'); pause(1.5)
    pyff('setint','MODE',4);
    pyff('set','BLOCK_SPEED',SPEED);
    pyff('set','DIST_STRESS_L',DIST_L);
    pyff('set','DIST_STRESS_H',DIST_H);
    pyff('setint','screen_pos',VP_SCREEN);
    
    fprintf('Press <RETURN> to start HandWorkStation Game.\n'); pause;
    fprintf('Ok, starting...\n'), close all
    
    pyff('play', 'basename', 'HandWorkStation', 'impedances',0);
    stimutil_waitForMarker(RUN_END);
    
    fprintf('HandWorkStation Game: run %d finished.\n',ii)
    pyff('quit'); pause(1);
    
end
