opt= [];
opt.disp_ival= {[-200 800], [-200 1200]};
opt.ref_ival= {[-200 0], [-200 0]};
opt.cfy_clab = {{'not','E*','Fp*','AF*','A*'}, ...
                {'not','E*','Fp*','AF*','A*'}};
opt.cfy_ival= {'auto', 'auto'};
opt.cfy_pick_peak= {[100 700], [100 1000]};
opt.model = 'RLDAshrink';
opt.ErrP_bias = 0;
opt.nr_sequences_threshold = 80; % min accuracy for selecting nr_sequences (in [%])
opt.nr_classes = 6;
opt.nhist = 50;
