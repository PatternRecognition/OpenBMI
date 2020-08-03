file_dir= [EEG_IMPORT_DIR 'bci_competition_iii/albany/'];

clab= {'FC5','FC3','FC1','FCz','FC2','FC4','FC6', ...
       'C5','C3','C1','Cz','C2','C4','C6', ...
       'CP5','CP3','CP1','CPz','CP2','CP4','CP6', ...
       'Fp1','Fpz','Fp2', 'AF7','AF3','AFz','AF4','AF8', ...
       'F7','F5','F3','F1','Fz','F2','F4','F6','F8', ...
       'FT7','FT8','T7','T8','T9','T10','TP7','TP8', ...
       'P7','P5','P3','P1','Pz','P2','P4','P6','P8', ...
       'PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'};


for Su= {'A','B'},
su= Su{1};

file= strcat(file_dir, 'Subject_', su, '_Train');
S= load(file);

clear cnt mrk
cnt.x= permute(double(S.Signal), [2 1 3]);
sz= size(cnt.x);
cnt.x= reshape(cnt.x, [sz(1)*sz(2) sz(3)]);
cnt.fs= 240;
cnt.clab= clab;
cnt.title= ['Data Set II - Subject ' su];

Flashing= double(S.Flashing');
Flashing= Flashing(:);
StimulusType= double(S.StimulusType');
StimulusType= StimulusType(:);
StimulusCode= double(S.StimulusCode');
StimulusCode= StimulusCode(:);
clear S

pos= find(diff([0; double(Flashing)])==1);
mrk.pos= pos';
mrk.toe= 2-StimulusType(pos)';
mrk.y= [mrk.toe==1; mrk.toe==2];
mrk.fs= cnt.fs;
mrk.className= {'deviant', 'standard'};
mrk.code= StimulusCode(pos)';
mrk.indexedByEpochs= {'code'};

mnt= projectElectrodePositions(cnt.clab);
grd= sprintf('F7,legend,Fz,scale,Fp2\nT7,CP3,CPz,CP4,T8\nTP7,P3,Pz,P4,TP8\nP7,PO3,POz,PO4,P8');
mnt= setDisplayMontage(mnt, grd, 'centerClab','Pz');
mnt= shrinkNonEEGchans(mnt);
mnt= setDisplayMontage(mnt, grd);

saveProcessedEEG([EEG_IMPORT_DIR 'bci_competition_iii/data_set_ii_' su], ...
                 cnt, mrk, mnt);

%% prepend 1s
cnt.x= cat(1, repmat(cnt.x(1,:), [cnt.fs 1]), cnt.x);
mrk.pos= mrk.pos + mrk.fs;

clear StimulusCode StimulusType Flashing epo

cnt1= proc_selectChannels(cnt, 1:32);
epo1= makeEpochs(cnt1, mrk, [-100 650]);
clear cnt1
epo1= proc_albanyAverageP3Trials(epo1, 15);
cnt2= proc_selectChannels(cnt, 33:64);
epo2= makeEpochs(cnt2, mrk, [-100 650]);
clear cnt cnt2
epo2= proc_albanyAverageP3Trials(epo2, 15);
epo= proc_appendChannels(epo1, epo2);

saveProcessedEEG([EEG_IMPORT_DIR 'bci_competition_iii/data_set_ii_' su], ...
                 epo, mrk, mnt, 'ave15');

end

