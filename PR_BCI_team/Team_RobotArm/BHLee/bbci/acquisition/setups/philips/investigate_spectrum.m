% - done:
%       * plot spec
%       * r2 plots for different bands
%       * r2 plots for 8-12, 8-10, 10-12 Hz (200ms slots)
%
% comment Anne: To Do:
%       * low pass filtern (40Hz)
%       * ERPs: Klassen 40-100Hz (class 1-7) in einen plot (P300 -> Cz)

force= 0;
opt_fig= strukt('folder', [TEX_DIR 'projects/project_philips/pics_auto/'], ...
    'format', 'pdf');

[sbj, session_name]= ...
    get_session_list('projects/project_philips','struct',1);

grd= sprintf(['_,F7,F3,Fz,F4,F8,_\n' ...
    'T7,C5,C3,Cz,C4,C6,T8\n' ...
    'A1,P7,P3,Pz,P4,P8,A2\n' ...
    '_,scale,O1,POz,O2,legend,_']);

opt_scalp= {'extrapolate', 0};
opt_grid= defopt_erps('scale_leftshift',0.075);
opt_grid_spec= defopt_spec('scale_leftshift',0.075, ...
    'xTickAxes','Pz');
opt_scalp_bp= defopt_scalp_power();
opt_scalp_erd= defopt_scalp_erp();
opt_scalp_r= defopt_scalp_r();

%%
for s=sbj

rootdir = '/home/bbci/data/bbciMat/';
file_name= [rootdir s.subdir '/philips' s.name];
opt_fig.prefix= [s.expid '_'];
fprintf('*** Processing %s\n', s.name);

%% default settings
stim_duration = 2000;
disp_ival= [0 stim_duration];

% to do: which time period to chose for reference?
%ref_ival= [0 100];
clab= 'Oz';
clab_rsq= {'Oz', 'Pz'};
clab_p3= {'Cz'};
crit_ival= [50 stim_duration];
crit_minmax= 110;

freq_range = 4;   % Frequency range around target frequency
ival_spec= [500 stim_duration];
ival_erd= [-1000 stim_duration+1000];
ival_list= [0 500; 500 1000; 1000 1500; 1500 2000];
% range [x Hz, y Hz] in which the spectrum is investigated
bad_clab= {};
reject_opts_ref= {'whiskerlength', 2, 'do_multipass',1};
reject_opts= {};
colAxIval= [-10 10];

%% load and preprocess data

hdr= eegfile_loadMatlab(file_name, 'vars','hdr');
ihighimp= find(max(hdr.impedances,[],1)>=50);
highimp_clab{s.num}= hdr.clab(ihighimp);
clab_load= cat(2, {'not','E*'}, bad_clab, highimp_clab{s.num});

% note: this does not includes all electrodes, only those with low
% impedances
% for changing this, uncomment the above line
clab_load= cat(2, {'not','E*'}, bad_clab);

[cnt, mrk, mnt, nfo]= eegfile_loadMatlab(file_name, 'clab', clab_load);
freqs = mrk.freq;
[mrk, rClab]= reject_varEventsAndChannels(cnt, mrk, disp_ival);

% [filt_b filt_a]= butter(5, [25 max_freq-1]/cnt.fs*2);
% cnt_flt= proc_filtfilt(cnt, filt_b, filt_a);

%% overview over the data
print_data_overview
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r=1;
bands = [freqs-r; freqs+r]';

if 1
    epo_spec= cntToEpo(cnt, mrk, ival_spec);
    
    epo_sf = epo_spec;
%     epo_sf = proc_laplacian(epo);
    %epo_sf = proc_commonAverageReference(epo_spec);
    
%     epo_CW = proc_selectClasses(epo_sf,13);
%     epo_80Hz = proc_selectClasses(epo_sf,5);
%     epo_80Hz_hits = proc_selectEpochs(epo_80Hz,find(epo_80Hz.flicker_seen));
%     epo_80Hz_hits.className{1} = '80Hz (hits)';
%     epo_80Hz_miss = proc_selectEpochs(epo_80Hz,find(epo_80Hz.flicker_seen==0));
%     epo_80Hz_miss.className{1} = '80Hz (miss)';
%     
%     epo_hits_CW = proc_appendEpochs(epo_80Hz_hits,epo_CW);
%     epo_miss_CW = proc_appendEpochs(epo_80Hz_miss,epo_CW);
%     epo_hits_miss = proc_appendEpochs(epo_80Hz_hits,epo_80Hz_miss);
%     epos = {epo_hits_CW, epo_miss_CW, epo_hits_miss};
%     
%     for n = 1:length(epos)
%         figure('Units','normalized','Position',[0 0 1 1]);
%         spec = proc_spectrum(epos{n}, bands(n,:), kaiser(cnt.fs,2));
%         spec.x = mean(spec.x,1);
%         spec = proc_r_square(spec);
%         scalpPlot(mnt, mean(spec.x,1));
%         title(cell2str(epos{n}.className,' - '))
%     end
%     keyboard
%     
%     % all r2 plots in one
%     fig_set(1)
%     for n = 1:length(epos)
%         subplot (1,3,n)
%         %figure('Units','normalized','Position',[0 0 1 1]);
%         spec = proc_spectrum(epos{n}, bands(n,:), kaiser(cnt.fs,2));
%         spec.x = mean(spec.x,1);
%         spec = proc_r_square(spec);
%         scalpPlot(mnt, mean(spec.x,1));
%         title(cellstr(epos{n}.className))
%         hold on
%     end
%     printFigure(['80Hz_r2'], [18 10], opt_fig, 'embed', 0);   
%     
    %% t-scale spectrum plots of all flicker frequencies
    warning off
    figure('Units','normalized');%,'Position',[0 0 1 1]);
    sc_pat = [];
    nFreqs = length(mrk.freq);
    for n = 1:nFreqs-1
%         subplot(2,ceil(nFreqs/2),n)
        subplot(1,nFreqs-1,n)
        epo = proc_selectClasses(epo_sf, [n length(epo_sf.className)]);
        %[filt_b filt_a]= butter(5, bands(n,:)/cnt.fs*2);
        %epo = proc_filtfilt(epo, filt_b, filt_a);
        %epo = proc_commonAverageReference(epo);
        spec = proc_spectrum(epo, bands(n,:), kaiser(cnt.fs,2));
        spec.x = mean(spec.x,1);
%         spec = proc_r_square(spec);1
        spec = proc_t_scale(spec,0.05);
        fprintf('Critical t-level for %s: %6.2f\n',epo_sf.className{n},spec.crit)
        sc_pat = [sc_pat; mean(spec.x,1)];
        h=scalpPlot(mnt, sc_pat(n,:), ...
            'colAx',[-2.5 2.5]*spec.crit,opt_scalp{:}, ...
            'contour',[-2 -1 -.5 .5 1 2]*spec.crit);
        set(h.contour,'LineWidth',2);
        title([epo_sf.className{n} ' [t-crit=' sprintf('%6.2f',spec.crit) ']'])
    end
    warning on
	printFigure(['t_flicker_freqs'], [30 18], opt_fig, 'embed', 0);
    
    %% r2 spectrum plots of all flicker frequencies (viualization normalized!)
    ma = max(max(sc_pat));
    figure('Units','normalized','Position',[0 0 1 1]);
    for n = 1:length(epo_sf.className)-1
        supplot(3,4,n)
        scalpPlot(mnt, sc_pat(n,:),'colAx',[0 ma]);
        title(epo_sf.className{n})
    end
    
else
    %% "ERD" plots of all flicker frequencies
    epo_spec= cntToEpo(cnt, mrk, [-2500 4500]);
    for fb = 1:length(epo_spec.className)-1
        % calculate ERD for the given frequency band
        [filt_b filt_a]= butter(5, bands(n,:)/cnt.fs*2);
        epo= proc_filtfilt(proc_selectClasses(epo_spec,[fb 13]), filt_b, filt_a);
        epo= proc_envelope(epo, 'ma_msec', 200);
        epo= proc_baseline(epo, [-500 0], 'trialwise', 1);
        epo= proc_selectIval(epo, [-2000 4000]);
        epo= proc_r_square(epo);
        % plot erd
        clab = 'PO8';
        ival_scalps_erd = [-1000 -500; -500 0; 0 250; 250 500; 500 1000; 1000 2000; 2000 2500; 2500 3000];
        figure('Units','normalized','Position',[0 0 1 1])
        scalpEvolutionPlusChannel(proc_average(epo), mnt, clab, ival_scalps_erd, ...
            defopt_scalp_erp, ...
            ... %'colorOrder', [0 1 0; 1 0 0; 0 0 0], ...
            'channelLineStyleOrder', {'thick','thin'}, ...
            'legend_pos','none');
    end
end

end


