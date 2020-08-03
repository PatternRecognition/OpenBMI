file= 'Seppel_01_05_23/dzweiSeppel';
%file= 'Stefan_01_05_10/dzweiStefan';
%file= 'Gabriel_00_11_03/dzweiGabriel';
%file= 'Frank_01_07_19/dzweiFrank';
%file= 'Motoaki_01_06_07/dzweiMotoaki';
%file= 'Steven_01_06_13/dzweiSteven';
%file= 'Thorsten_01_05_15/dzweiThorsten';
%file= 'Ben_01_05_31/dzweiBen';

%% see also .../bci/papers/icann02/

[cnt, mrk, mnt]= loadProcessedEEG(file);
mrk= getDzweiEvents(file);

epo= makeSegments(cnt, mrk, [-100 800]);
epo= proc_baseline(epo, [-100 0]);
epo= proc_rectifyChannels(epo, 'EMGl', 'EMGr');
epo= proc_baseline(epo, [-100 0]);

showERPgrid(epo, mnt);
