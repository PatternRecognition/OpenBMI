subdir= TODAY_DIR;

grd= sprintf(['scale,F5,F3,F1,Fz,F2,F4,F6,legend\n' ...
    'FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8\n' ...
    'T7,C5,C3,C1,Cz,C2,C4,C6,T8\n' ...
    'TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8\n' ...
    'P5,P3,PO3,O1,Pz,O2,PO4,P4,P6\n']);
opt_grid_spec= defopt_spec('scale_leftshift',0.075, ...
    'xTickAxes','PO4');
opt_grid_spec.colorOrder= [1 0 1; 0 0.7 0; 0 0 0];
opt_scalp_bp= defopt_scalp_power('resolution',30, ...
    'extrapolate', 1);
opt_scalp_bp.colorOrder= opt_grid_spec.colorOrder;
opt_scalp_r= defopt_scalp_r('resolution', 30, ...
    'extrapolate', 1);

    subdir= subdir_list{vp}
    sbj= subdir(1:find(subdir=='_',1,'first')-1);
    clab= 'Pz';
    band_list= [4 7.5; 7.5 9.5; 9.5 13.5; 13.5 35.5];
    reject_opts= {};
    
    file= [subdir '/relax' sbj];
    [cnt, mrk]= eegfile_loadMatlab(file);
    cnt= proc_selectChannels(cnt, 'not','Fp*');
    blk1= blk_segmentsFromMarkers(mrk, ...
        'start_marker','eyes_closed', ...
        'end_marker','stop');
    blk2= blk_segmentsFromMarkers(mrk, ...
        'start_marker','eyes_open', ...
        'end_marker','stop');
    blk= blk_merge(blk1, blk2, 'className',{'eyes closed','eyes open'});
    
    [cnt, blkcnt]= proc_concatBlocks(cnt, blk);
    mkk= mrk_evenlyInBlocks(blkcnt, 1000);
    cnt.title= sbj;
    
    fig_set(1);
    ct= proc_selectChannels(cnt, scalpChannels); %% order nicely
    [mkk, rClab]= reject_varEventsAndChannels(ct, mkk, [0 999], ...
        'visualize', 1, ...
        reject_opts{:});
    clear ct;
    mnt= getElectrodePositions(cnt.clab);
    mnt= mnt_setGrid(mnt, grd);
    ii= chanind(mnt, 'PO*');
    mnt.box(2,ii)= mnt.box(2,ii)-0.15;
    ii= chanind(mnt, 'O*');
    mnt.box(2,ii)= mnt.box(2,ii)-0.3;
    mntlap= mnt;
    mnt_red= mnt_restrictMontage(mnt, 'not', rClab);
    
    spec= cntToEpo(cnt, mkk, [0 1000], 'mtsp', 'before');
    spec_lap= proc_localAverageReference(spec, mnt, 'radius',0.6, 'verbose',vp==2);
    spec_lap= proc_spectrum(spec_lap, [1 40], kaiser(cnt.fs,2));
    spec= proc_spectrum(spec, [1 40], kaiser(cnt.fs,2));
    spec_r= proc_r_square_signed(proc_selectClasses(spec, [1 2]));
    spec_lap_r= proc_r_square_signed(proc_selectClasses(spec_lap, [1 2]));
    
    H= grid_plot(spec, mnt, opt_grid_spec);
    grid_addBars(spec_r, 'h_scale',H.scale);
    
    H= grid_plot(spec_lap, mntlap, opt_grid_spec);
    grid_addBars(spec_lap_r, 'h_scale',H.scale);
    
    fig_set(2);
    H= scalpEvolutionPlusChannel(spec, mnt_red, clab, band_list, ...
        opt_scalp_bp, ...
        'lineWidth', 2, ...
        'scalePos','horiz', ...
        'globalCLim',0, ...
        'legend_pos',1);
    grid_addBars(proc_rectifyChannels(spec_r), ...
        'cLim', '0tomax', ...
        'vpos',1);
    printFigure(['spec_topo'], [24 4+5.8*size(spec.y,1)], opt_fig);
    
    fig_set(4, 'shrink',[1 2/3]);
    spec_r.className= {sprintf('\\pm r^2 (EC,EO)')};
    scalpEvolutionPlusChannel(spec_r, mnt, clab, band_list, ...
        opt_scalp_r, ...
        'channelAtBottom',1, 'legend_pos',1);
    shiftAxesUp;
    clear spec spec_r spec_lap spec_lap_r
