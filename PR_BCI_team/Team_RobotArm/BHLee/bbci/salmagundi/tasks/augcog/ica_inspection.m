setup_augcog;
%nn= 4;
%task= 'auditory';
%condition= 'low';

fig_dir= 'augcog_misc/';

classDef= {'D*','S*';'D','S'};
blk= getAugCogBlocks(augcog(nn).file);
blk= blk_selectBlocks(blk, [condition ' ' task]);
[cnt, bmrk, Mrk]= readBlocks(augcog(nn).file, blk, classDef);

mnt= projectElectrodePositions(cnt.clab);
mnt.y= 1.2*mnt.y;
opt= struct('colorOrder', [1 0 0; 0 0.7 0]);
opt.xGrid= 'on';
opt.yZeroLine= 'on';
spec_opt= opt;
spec_opt.xTick= 10:10:40;
spec_opt.xTickLabel= '';
opt.resolution= 32;
opt.shading= 'flat';

mrk= Mrk;

%use only EEG channels (i.e. channels with position on the scalp)
chans= find(~isnan(mnt.x));
cnt= proc_selectChannels(cnt, chans);

%apply some high-pass filter
cnt= proc_subtractMovingAverage(cnt, 3000);

%apply TDsep
tau= 0:5;
W= tdsep0(cnt.x(:,:)', tau);

epo= makeEpochs(cnt, mrk, [-200 800]);
epo= proc_baseline(epo, [-200 0]);

close all
plotPatternsPlusSpecERP(epo, mnt, W, opt, 'spec_opt',spec_opt);
%plotPatternsPlusERP(epo, mnt, W, opt);
%plotPatternsPlusSpec(epo, mnt, W, opt);

%% show a selection of artifact components
opt.selection= [1 4 24 6];
%% for printing
opt.fontSize= 10;
spec_opt.fontSize= 10;
plotPatternsPlusSpecERP(epo, mnt, W, opt, 'spec_opt',spec_opt);
saveFigure([fig_dir exp_name '_ica_artifacts_' task '_' condition], [10 5]*2);

%% show a selection of cortical sources
opt.selection= [12 20 25 27];
plotPatternsPlusSpecERP(epo, mnt, W, opt, 'spec_opt',spec_opt);
saveFigure([fig_dir exp_name '_ica_physiol_' task '_' condition], [10 5]*2);
