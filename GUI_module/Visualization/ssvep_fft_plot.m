function ssvep_fft_plot(filepath,subject_info, filename)

file=fullfile(filepath,subject_info.subject,subject_info.session, filename);
marker={'1','up';'2','left'; '3', 'right'; '4', 'down'};
fs=200; 
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',fs});

cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
cnt=prep_filter(cnt, {'frequency', [5 40]});
smt=prep_segmentation(cnt, {'interval', [0 4000]});
visual_fft(smt,{'channel','Oz';'xlim',[3 20];'plot','on';'line',true;'filepath',filepath;'subject_info',subject_info;'filename',filename});