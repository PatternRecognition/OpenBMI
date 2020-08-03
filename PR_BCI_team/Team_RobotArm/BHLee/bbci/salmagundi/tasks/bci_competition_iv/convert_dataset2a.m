DATDIR= '/home/blanker/data/bci_competition_iv/dataset_2a/';

cd('~/matlab/Import');
biosig_installer

file_list= cprintf('A%02dT.gdf', 1:9);


for vp= 1:length(file_list),

[s,h]= sload([DATDIR file_list{vp}]);

cnt= [];
cnt.x= s;
cnt.fs= h.SampleRate;
cnt.clab= {'Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz','EOGl','EOGc','EOGr'};
clear s

mrk= [];
mrk.pos= h.EVENT.POS';
mrk.fs= h.EVENT.SampleRate;
mrk.y= [h.EVENT.TYP'==769;
        h.EVENT.TYP'==770;
        h.EVENT.TYP'==771;
        h.EVENT.TYP'==772];
mrk.className= {'left', 'right', 'foot', 'tongue'};
valid= find(any(mrk.y,1));
mrk= mrk_chooseEvents(mrk, valid);

grd= sprintf(['EOGl,_,_,Fz,_,EOGc,EOGr\n' ...
              '_,FC3,FC1,FCz,FC2,FC4,_\n' ...
              'C5,C3,C1,Cz,C2,C4,C6\n' ...
              '_,CP3,CP1,CPz,CP2,CP4,_\n' ...
              '_,_,P1,Pz,P2,_,_\n' ...
              'scale,_,_,POz,_,_,legend']);
mnt= getElectrodePositions(cnt.clab);
mnt= mnt_setGrid(mnt, grd);

filename= [DATA_DIR 'egMat/bci_competition_iv/' file_list{vp}];
eegfile_saveMatlab(filename, ...
                   cnt, mrk, mnt);

end
