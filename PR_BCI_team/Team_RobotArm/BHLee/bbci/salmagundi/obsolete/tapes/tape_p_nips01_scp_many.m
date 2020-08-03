fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 2], 128, 150);
fv= proc_subsampleByMean(fv, 5);

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 150);
fv= proc_subsampleByMean(fv, 5);

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 5], 128, 150);
fv= proc_subsampleByMean(fv, 5);

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 5], 128, 150);
fv= proc_subsampleByMean(fv, 4);

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 2], 128, 200);
fv= proc_subsampleByMean(fv, 5);

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200);
fv= proc_subsampleByMean(fv, 5);

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 5], 128, 200);
fv= proc_subsampleByMean(fv, 5);

fv= proc_selectChannels(epo, 'F#', 'FC#', 'C#', 'CP#', 'O1');
fv= proc_filtBruteFFT(fv, [0.8 2], 128, 150);
fv= proc_subsampleByMean(fv, 5);

fv= proc_selectChannels(epo, 'F#', 'FC#', 'C#', 'CP#', 'O1');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 150);
fv= proc_subsampleByMean(fv, 5);

fv= proc_selectChannels(epo, 'F#', 'FC#', 'C#', 'CP#', 'O1');
fv= proc_filtBruteFFT(fv, [0.8 5], 128, 150);
fv= proc_subsampleByMean(fv, 5);
