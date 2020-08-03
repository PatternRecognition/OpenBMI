function fv=visualQuiz_feat_extract(fv,varargin)

opt= propertylist2struct(varargin{:});
[opt, dummy]= ...
    set_defaults(opt, ...        
                 'nIvals', 5, ...
                 'ival',[-2000 18000], ...
                 'ref_ival',[-2000 0], ...
                 'signal','deoxy', ...
                 'lowpass', 1, ...
                 'innerFold',0, ...
                 'baseline',0,...
                 'justFront',0,...
                 'clab',{}, ...
                 'justBack',0);
             
% just front? or just back?
front_ch={'19_20','19_37','19_Gnd','19_38','1_Ref','1_Gnd','1_20','10_77', ...
              '1_21','1_78','76_GND','76_77','76_94','76_95','38_37','38_20',...
              '38_21','38_22','38_39','38_53','2_21','2_78','2_22','2_3', ...
              '2_80','2_79','95_77','95_94','95_110','95_96','95_79','95_78', ...
              '95_78','23_22','23_3','23_39','80_79','80_3','80_96'};
          
back_ch={'47_46','28_8','28_9','85_8','85_103','85_9','104_103','104_87', ...
         '12_47','12_30','12_11','29_30','29_46','29_9','86_9','86_103', ...
              '86_87','29_11','29_10','86_10','86_68','69_68','69_87','87_104'};
if opt.justFront
    fv=proc_selectChannels(fv,front_ch);
elseif opt.justBack
    fv=proc_selectChannels(fv,back_ch);
end

% Baseline 
if opt.baseline   
  fv = proc_baseline(fv,opt.ref_ival);
end

% Select post-stimulus interval
% fv = proc_selectIval(fv,[opt.ref_ival(2) opt.ival(2)]);


%% Select features
if opt.innerFold
    allclab = cell2mat(strcat('''',fv.clab,''','));
    allclab = allclab(1:end-1);
    proc.train = ['fvr = proc_r_square_signed(fv);',...
    '[ival_cfy,nfo]= select_time_intervals(fvr,''nIvals'',',num2str(opt.nIvals),',''score_factor_for_max'',2,''clab_pick_peak'', {' allclab '});',...
     'fv= proc_jumpingMeans(fv, ival_cfy);'];

    proc.apply= ['x_orig = sum(fv.x,1); fv= proc_jumpingMeans(fv, ival_cfy);'];

    proc.memo = {'ival_cfy'};
else
    fvr = proc_r_square_signed(fv);
    warning off
    [ival_cfy,nfo]= select_time_intervals(fvr,'nIvals',opt.nIvals,'clab_pick_peak', fv.clab);
    warning on
    ival_cfy
 
    fv= proc_jumpingMeans(fv, ival_cfy);
end


         
                 