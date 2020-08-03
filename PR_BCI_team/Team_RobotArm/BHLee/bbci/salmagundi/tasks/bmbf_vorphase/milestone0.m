picsDir= '/home/nibbler/blanker/candy/Tex/bci/pics/bmbf_vorphase/';


file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file, 'cut50');
mnt= setDisplayMontage(mnt, 'selfpaced32');
nonEEGchans= chanind(mnt, 'E*');
mnt.box(:,nonEEGchans)= mnt.box(:,nonEEGchans) + ...
    0.1*sign(mnt.box(:,nonEEGchans));
epo= makeSegments(cnt, mrk, [-1200 600]);
epo= proc_baseline(epo, [-1200 -800]);
epo= proc_rectifyChannels(epo, 'EMGl', 'EMGr');
epo= proc_baseline(epo, [-1200 -800]);
ht= showERPgrid(epo, mnt, [-15 15]);
delete(ht)

set(gcf, 'paperUnits','centimeters', ...
         'paperPosition', [0 0 10 5]*3);
print('-dpsc2', [picsDir 'fig0a_ERP2s']);

epo_lap= proc_laplace(epo, 'small');
mnt_lap= setDisplayMontage(mnt, 'C3C4');
mnt_lap= adaptMontage(mnt_lap, epo_lap, ' lap');
ht= showERPgrid(epo_lap, mnt_lap, [-5 5]);
delete(ht);
set(gcf, 'paperPosition', [0 0 10 8]);
print('-dpsc2', [picsDir 'fig0b_ERP2s_lap']);

ht= showERtvalueGrid(epo_lap, mnt_lap, 0.01, [-25 25]);
delete(ht);
set(gcf, 'paperPosition', [0 0 10 8]);
print('-dpsc2', [picsDir 'fig0c_ERP2s_lap_tvalues']);



%file= 'Gabriel_01_07_24/selfpaced0_5sGabriel';
file= 'Seppel_01_10_19/selfpaced0_5sSeppel';

[cnt, mrk, mnt]= loadProcessedEEG(file, 'cut50');
mnt= setDisplayMontage(mnt, 'selfpaced32');
[pairs, pace, sy]= getEventPairs(mrk, 1000);
equi= equiSubset(pairs);
mrk.y(:)= 0;
mrk.y(1,[equi{1} equi{2}])= 1;
mrk.y(2,[equi{3} equi{4}])= 1;

epo= makeSegments(cnt, mrk, [-400 200]);
epo= proc_baseline(epo, [-400 -300]);
epo= proc_rectifyChannels(epo, 'EMGl', 'EMGr');
epo= proc_baseline(epo, [-400 -300]);
epo_lap= proc_laplace(epo, 'small');
mnt_lap= setDisplayMontage(mnt, 'C3C4');
mnt_lap= adaptMontage(mnt_lap, epo_lap, ' lap');
ht= showERPgrid(epo_lap, mnt_lap, [-1 1]);
delete(ht);
set(gcf, 'paperPosition', [0 0 10 8]);
print('-dpsc2', [picsDir 'fig0d_ERP0_5s_lap']);

ht= showERtvalueGrid(epo_lap, mnt_lap, 0.01, [-10 10]);
delete(ht);
set(gcf, 'paperPosition', [0 0 10 8]);
print('-dpsc2', [picsDir 'fig0e_ERP0_5s_lap_tvalues']);
