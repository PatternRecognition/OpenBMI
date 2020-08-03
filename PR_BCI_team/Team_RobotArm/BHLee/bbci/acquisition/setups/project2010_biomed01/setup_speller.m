fbint= struct();
fbint.screenPos= VP_SCREEN;
%fbint.fullscreen= 1;
fbint.nr_sequences= 5;

pyff('init', 'CenterSpellerTeam02');
pyff('load_settings', [BCI_DIR 'acquisition/setups/project2010_biomed01/CenterSpeller_default']);
pyff('setint', fbint);
