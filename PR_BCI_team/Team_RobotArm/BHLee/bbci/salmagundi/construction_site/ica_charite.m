%file= 'charite/Andrea290502_mu'; chan_list= {7, 17, 27, 31};
file= 'charite/Fabian_100702_mu'; chan_list= {7, 23, 26, 27};
%file= 'charite/Petra_011002_ech'; chan_list= {8};
%file= 'charite/gerrit_block_mu'; chan_list= {6, 7, 17, 21};
%file= 'charite/gerrit_ec_tast'; chan_list= {6, 8, 26, 27};

fs= 100;
cnt= readGenericEEG(file, [], fs);
cnt= proc_selectChannels(cnt, 'not','SAP');
classDef= {'l', 'r'; 'left','right'};
mrk= readMarkerComments(file, fs, classDef);
mnt= setElectrodeMontage(cnt.clab);

%% perform source separation on the first recording segment
tau= 0:5;
tic; W= tdsep0(cnt.x', tau); toc


%% show scalp patterns of ICs
A= inv(W);
nICs= size(W,1);
opt.resolution= 32;
opt.shading= 'flat';
opt.scalePos= 'none';
opt.colAx= 'range';
for ic= 1:nICs,  suplot(nICs, ic, .01, .01);
  plotScalpPattern(mnt, A(:,ic), opt);
  hl= ylabel(sprintf('ic %d', ic)); set(hl,'visible','on');
end
%addTitle(cnt.title);
%saveFigure([fig_dir 'ica_patterns'], [10 6]*2);


%% apply unmixing matrix to data
cnt_ic= copyStruct(cnt, 'x');
cnt_ic.x= cnt.x * W';
for ci= 1:length(cnt_ic.clab),
  cnt_ic.clab{ci}= sprintf('ic%d', ci);
end

%% show spectra of continuous ICs
spec= proc_spectrum(cnt_ic, [4 34]);
clf;
for ic= 1:length(chan_list),
  suplot(length(chan_list), ic, [0.1 0.12 0.08], [0.1 0.05 0.05]);
  showERP(spec, mnt, chan_list{ic});
%  xlabel('[Hz]');
  grid on;
end
unifyYLim;
%saveFigure([fig_dir 'ica_spec'], [10 6]*2);


opt.resolution= 40;
opt.contour= 1;
nChans= length(chan_list);
for ii= 1:nChans,
  ic= chanind(epo, chan_list{ii});
  suplot(nChans, ii, .05, .05);
  showScalpPattern(mnt, A(:,ic), opt);
  hl= ylabel(sprintf('ic %d', ic)); set(hl,'visible','on');
end
%addTitle(cnt.title);
%saveFigure([fig_dir 'ica_patterns_selected'], [10 6]*2);


if ~isempty(mrk.pos),
  epo= makeSegments(cnt_ic, mrk, [1 17]*1000);
  spec= proc_spectrum(epo, [4 34], epo.fs);
  clf;
  for ic= 1:length(chan_list),
    suplot(length(chan_list), ic, [0.1 0.12 0.08], [0.1 0.05 0.05]);
    showERP(spec, mnt, chan_list{ic});
    legend(epo.className);
    grid on;
  end
  unifyYLim; 
%  saveFigure([fig_dir 'ica_epo_spec'], [10 6]*2);
end
        