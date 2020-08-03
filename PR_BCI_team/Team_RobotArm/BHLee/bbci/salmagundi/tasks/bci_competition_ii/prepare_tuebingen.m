data_dir= [DATA_DIR 'eegImport/bci_competition_ii/tuebingen/'];


%
% a34lk training data
%

base_name= 'a34lkt/Traindata';

epo= struct('fs',256, 'className',{{'neg','pos'}});
epo.clab= {'A1mCz', 'A2mCz', 'C3a', 'C3p', 'C4a', 'C4p'};
file= sprintf('%s%s_0.txt', data_dir, base_name);
epo.x= textread(file, '');
file= sprintf('%s%s_1.txt', data_dir, base_name);
dat= textread(file, '');
epo.x= cat(1, epo.x, dat)';
clear dat

epo.y= epo.x(1,:);
epo.y= [epo.y==0; epo.y==1];
epo.x= epo.x(2:end-1,:);
nChans= length(epo.clab);
T= size(epo.x,1)/nChans;
epo.x= reshape(epo.x, [T nChans size(epo.x,2)]);
%epo.t= linspace(2000, 5500, T+1);
%epo.t= epo.t(1:end-1);
epo.t= linspace(2000, 5500, T);
epo.title= 'tuebingen a34lkt';
            
%% add TTD channel
ttd_chans= chanind(epo, 'A1mCz', 'A2mCz');
ttd_weights= [-0.5 -0.5]';
sz= size(epo.x);
epo.x= reshape(permute(epo.x, [1 3 2]), sz(1)*sz(3),sz(2));;
epo.x= cat(2, epo.x, epo.x(:,ttd_chans) * ttd_weights);
epo.x= permute(reshape(epo.x, [sz(1) sz(3) sz(2)+1]), [1 3 2]);
epo.clab= cat(2, epo.clab, {'TTD'});

mnt= setElectrodeMontage(epo.clab);
grd= sprintf('C3a,A1mCz,C4a\nC3p,A2mCz,C4p,\nTTD,_,legend');
mnt= setDisplayMontage(mnt, grd);
mnt.box_sz(1,chanind(mnt,'TTD'))= 2*mnt.box_sz(1,chanind(mnt,'TTD'));

saveProcessedEEG('bci_competition_ii/tuebingen1_train', epo, [], mnt);


%
% a34lk test data
%

base_name= 'a34lkt/Testdata';

epo= struct('fs',256, 'className',{{'neg','pos'}});
epo.clab= {'A1mCz', 'A2mCz', 'C3a', 'C3p', 'C4a', 'C4p'};
file= sprintf('%s%s.txt', data_dir, base_name);
epo.x= textread(file, '')';
epo.x= epo.x(1:end-1,:);

nChans= length(epo.clab);
T= size(epo.x,1)/nChans;
epo.x= reshape(epo.x, [T nChans size(epo.x,2)]);
epo.y= zeros(2,size(epo.x,3));
epo.t= linspace(2000, 5500, T);
epo.title= 'tuebingen a34lkt test data';

%% add TTD channel
ttd_chans= chanind(epo, 'A1mCz', 'A2mCz');
ttd_weights= [-0.5 -0.5]';
sz= size(epo.x);
epo.x= reshape(permute(epo.x, [1 3 2]), sz(1)*sz(3),sz(2));;
epo.x= cat(2, epo.x, epo.x(:,ttd_chans) * ttd_weights);
epo.x= permute(reshape(epo.x, [sz(1) sz(3) sz(2)+1]), [1 3 2]);
epo.clab= cat(2, epo.clab, {'TTD'});

saveProcessedEEG('bci_competition_ii/tuebingen1_test', epo, [], mnt);


%
% egl2ln training data
%

base_name= 'egl2ln/Traindata';

epo= struct('fs',256, 'className',{{'neg','pos'}});
epo.clab= {'A1mCz', 'A2mCz', 'C3a', 'C3p', 'EOGv',  'C4a', 'C4p'};
file= sprintf('%s%s_0.txt', data_dir, base_name);
epo.x= textread(file, '');
file= sprintf('%s%s_1.txt', data_dir, base_name);
dat= textread(file, '');
epo.x= cat(1, epo.x, dat)';
clear dat

epo.y= epo.x(1,:);
epo.y= [epo.y==0; epo.y==1];
epo.x= epo.x(2:end-1,:);
nChans= length(epo.clab);
T= size(epo.x,1)/nChans;
epo.x= reshape(epo.x, [T nChans size(epo.x,2)]);
%epo.t= linspace(2000, 6500, T+1);
%epo.t= epo.t(1:end-1);
epo.t= linspace(2000, 6500, T);
epo.title= 'tuebingen egl2ln';

%% add TTD channel
ttd_chans= chanind(epo, 'A1mCz', 'A2mCz', 'EOGv');
ttd_weights= [-0.5 -0.5 -0.12]';
sz= size(epo.x);
epo.x= reshape(permute(epo.x, [1 3 2]), sz(1)*sz(3),sz(2));;
epo.x= cat(2, epo.x, epo.x(:,ttd_chans) * ttd_weights);
epo.x= permute(reshape(epo.x, [sz(1) sz(3) sz(2)+1]), [1 3 2]);
epo.clab= cat(2, epo.clab, {'TTD'});

mnt= setElectrodeMontage(epo.clab);
grd= sprintf('C3a,A1mCz,C4a\nC3p,A2mCz,C4p,\nTTD,legend,EOGv');
mnt= setDisplayMontage(mnt, grd);

saveProcessedEEG('bci_competition_ii/tuebingen2_train', epo, [], mnt);


%
% egl2ln test data
%

base_name= 'egl2ln/Testdata';

epo= struct('fs',256, 'className',{{'neg','pos'}});
epo.clab= {'A1mCz', 'A2mCz', 'C3a', 'C3p', 'EOGv',  'C4a', 'C4p'};
file= sprintf('%s%s.txt', data_dir, base_name);
epo.x= textread(file, '')';
epo.x= epo.x(1:end-1,:);

nChans= length(epo.clab);
T= size(epo.x,1)/nChans;
epo.x= reshape(epo.x, [T nChans size(epo.x,2)]);
epo.y= zeros(2,size(epo.x,3));
epo.t= linspace(2000, 5500, T);
epo.title= 'tuebingen egl2ln test data';

%% add TTD channel
ttd_chans= chanind(epo, 'A1mCz', 'A2mCz', 'EOGv');
ttd_weights= [-0.5 -0.5 -0.12]';
sz= size(epo.x);
epo.x= reshape(permute(epo.x, [1 3 2]), sz(1)*sz(3),sz(2));;
epo.x= cat(2, epo.x, epo.x(:,ttd_chans) * ttd_weights);
epo.x= permute(reshape(epo.x, [sz(1) sz(3) sz(2)+1]), [1 3 2]);
epo.clab= cat(2, epo.clab, {'TTD'});

saveProcessedEEG('bci_competition_ii/tuebingen2_test', epo, [], mnt);
