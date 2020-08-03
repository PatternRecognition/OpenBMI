data_dir= [DATA_DIR 'eegImport/bci_competition_ii/graz/'];

file= 'graz_data';
load([data_dir file]);

clear epo
epo.x= x_train;
epo.y= [y_train'==1; y_train'==2];
epo.t= linspace(0, 9000, size(epo.x,1));
epo.clab= {'C3 bip', 'Cz bip', 'C4 bip'};
epo.fs= 128;
epo.className= {'left','right'};
epo.title= 'graz';
clear x_train y_train 

mnt= setElectrodeMontage(strhead(epo.clab));
grd= sprintf('C3,C4\nCz,legend');
mnt= setDisplayMontage(mnt, grd);

saveProcessedEEG('bci_competition_ii/graz_train', epo, [], mnt);


epo.x= x_test;
epo.y= zeros(2,size(epo.x,3));
epo.title= 'graz test data';
saveProcessedEEG('bci_competition_ii/graz_test', epo, [], mnt);
