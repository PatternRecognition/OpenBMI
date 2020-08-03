fv= proc_selectChannels(epo, 'F#', 'FC3-4', 'C3-4', 'CP3-4', 'P3','P4');
fv= proc_selectIval(fv, [0 350]);
fv= proc_jumpingMeans(fv, 5);

fv= proc_selectChannels(epo, 'F#','FC#','C#','CP#');
fv= proc_selectIval(fv, [0 100]);
fv= proc_jumpingMeans(fv, 5);

fv= proc_selectChannels(epo, 'F#','FC#','C#','CP#','P3','P4');
fv= proc_selectIval(fv, [150 350]);
fv= proc_jumpingMeans(fv, 5);

fv= proc_selectChannels(epo, 'F#', 'FC3-4', 'C3-4', 'CP3-4', 'P3','P4');
fv= proc_selectIval(fv, [0 300]);
fv= proc_jumpingMeans(fv, 5);

fv= proc_selectChannels(epo, 'F#', 'FC3-4', 'C3-4', 'CP3-4', 'P3','P4');
fv= proc_selectIval(fv, [0 250]);
fv= proc_jumpingMeans(fv, 5);
