file= 'Gabriel_01_07_24/selfpaced1sGabriel';
save_file= '/home/neuro/data/dropbox/blanker/selfpaced1s_010724_1000Hz';

[cnt,mk]= eegfile_loadBV(file)

classDef= {[65 70], [74 192]; 'left','right'};
mrk= mrk_defineClasses(mk, classDef);

save(save_file, 'cnt', 'mrk');
