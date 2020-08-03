%% Settings for pyff oddball (P300_rectangle)
fb= [];
fb.screenPos = int16(VP_SCREEN);
fb.stim_duration = 1500;
fb.nStim = 200;
fb.nStim_per_block = 100;

pyff('init','P300_Rectangle')
pyff('set',fb)

