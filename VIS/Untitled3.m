load('D:\cell_epoch');
for i = 1:2
    dat = epos_av{i};
    dat.chan = dat.clab;
    dat.ival = dat.t;
    dat.x = permute(dat.x, [1 3 2]);
    dat.se = permute(dat.se, [1 3 2]);
    dat.class = {'1','target';'2','nontarget'};
    dat = rmfield(dat, {'y', 'event', 'mrk_info', 'cnt_info', 'history', 'refIval', 'p', 'tstat', 'sgnlogp', 'N', 'df', 'indexedByEpochs', 'clab', 't', 'className'});
    epos_av{i} = dat;
end

av_SMT = grandAverage_prototye(epos_av);


%%
load('D:\cell_rval');
for i = 1:2
    dat = epos_r{i};
    dat.chan = dat.clab;
    dat.ival = dat.t;
    dat.x = permute(dat.x, [1 3 2]);
    dat.se = permute(dat.se, [1 3 2]);
    dat.class = {'1','sgnr^2'};
    dat = rmfield(dat, {'y', 'event', 'mrk_info', 'cnt_info', 'history', 'refIval', 'p', 'sgnlogp', 'clab', 't', 'className'});
    epos_r{i} = dat;
end

av_SMT = grandAverage_prototye(epos_r);

%% 비교
SMT = prep_addTrials(MI_cell);
SMT = prep_envelope(SMT);
SMT = prep_baseline(SMT, {'Time', [-500 0]});
% SMT = proc_signedrSquare(SMT);
SMT = prep_average(SMT);

a = reshape(SMT.x, [], 1);
b = reshape(av_SMT.x, [], 1);

fprintf('a: %f\nb: %f\n', a(223), b(223));

%
%% ERP
load('D:\GIga_data\MI_all');

for i = 1:length(MI_cell)
    dat = MI_cell{i};
    SMT = prep_envelope(dat);
    SMT = prep_baseline(SMT, {'Time', [-500 0]});
    avSMT = prep_average(SMT);
    rSMT = proc_signedrSquare(SMT);
    
    avCell{i} = avSMT;
    rCell{i} = rSMT;
end

av_SMT = grandAverage_prototye(avCell);
av_SMT = grandAverage_prototye(avCell);

interval = [1000 1300; 1500 1800; 2000 2300;];
channels = {'C3', 'C4'};
classes = {'right'; 'left'};
TimePlot = 'on'; TopoPlot = 'on'; rValue = 'on'; MIPlot = 'on';
p_range = 'mean'; cm = 'parula'; patcher = 'on';
quality = 'high'; Align = 'vert'; baseline = [-1000 0];

options = {'Interval', interval; 'Channels', channels; 'Class', classes;...
    'TimePlot', 'on'; 'TopoPlot', 'on'; 'rValue', 'on'; 'MIPlot', 'on';...
    'Range', p_range; 'Colormap', cm; 'Patch', patcher; 'Quality', quality;
    'Align', Align; 'baseline', baseline};

vis_plotCont2(av_SMT, r_SMTptions);

%% 비교
SMT = prep_addTrials(MI_cell);
SMT = prep_envelope(SMT);
SMT = prep_baseline(SMT, {'Time', [-500 0]});
av_SMT = prep_average(SMT);
r_SMT = proc_signedrSquare(SMT);

vis_plotCont2(av_SMT, r_SMT, options);