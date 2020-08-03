acqFolder = [BCI_DIR 'acquisition/setups/labrotation2011_MirkoDietrich/'];

pyff('init', 'CenterDotsCakeSpellerMVEP');
pause(1);
pyff('load_settings', [acqFolder 'CenterDotsCakeSpellerMVEP_default']);
pause(1);
pyff('setint', 'screenPos', VP_SCREEN);
