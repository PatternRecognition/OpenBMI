fbint= struct();
REDUCE= 500;
fbint.geometry= [REDUCE/2 0 1280-REDUCE 1024];
%fbint.geometry= [1440+REDUCE/2 0 1280-REDUCE 1024];
%fbint.geometry= [200 0 1520 1200];
%fbint.geometry= [0 0 640 480];
fbint.fullscreen= 0;
fbint.nr_sequences= 5;
fbint.matrix_font_size= 100;

pyff('init', 'P300Matrix');
pause(1);
pyff('load_settings', [BCI_DIR 'acquisition/setups/project2010_biomed02/MatrixSpeller_default']);
pyff('setint', fbint);
