
% <<<step4>>> converting the format of files from mat to set
% (updating later)
 
EEG = pop_importdata('dataformat', 'matlab', 'nbchan', 56, 'data', 'D:\anedata_step\d2_MM_1.mat', 'srate', 200, 'pnts', 0, 'xmin', 0);
EEG = pop_importevent(EEG, 'event', 'D:\practice\MM_1_event.txt','fields', {'type' 'latency' 'duration'}, 'skipline', 1, 'timeunit', 1);
EEG = pop_chanedit(EEG, 'lookup','C:\Users\cvpr\Desktop\code_ane\eeglab13_4_4b\plugins\dipfit2.3\standard_BESA\standard-10-5-cap385.elp','load',{'D:\mjlee_cb\56chansCED.ced' 'filetype' 'autodetect'});
EEG = pop_saveset( EEG, 'filename','MM_1','filepath','D:\practice');
