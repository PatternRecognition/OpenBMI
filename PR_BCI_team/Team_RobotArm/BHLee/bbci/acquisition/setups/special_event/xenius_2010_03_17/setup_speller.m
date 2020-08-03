acqFolder = [BCI_DIR 'acquisition/setups/special_event/xenius_2010_03_17/'];

pyff('init', [speller_name 'VE']);

pause(1);
pyff('load_settings', [acqFolder speller_name]);
pause(1);
pyff('setint', 'screenPos', VP_SCREEN);
pause(0.1)
