dp= diff(mrk.pos/mrk.fs*1000);
idel= 1+find(dp<=2000);
mrk= mrk_chooseEvents(mrk, 'not',idel);
%% extract target markers during run 1:
mrk_r1= mrk_selectEvents(mrk, find(mrk.pos<=Cnt.T(1)));
%% extract target markers during primary task in run2:
blk_ls= blk_segmentsFromMarkers(mrk_orig, ...
    'start_marker','S 41','end_marker','S 40', ...
    'skip_unfinished',0);
blk_ls.ival(2,end)= sum(Cnt.T);
% correct for transition between tasks
blk_ls.ival(1,:)= blk_ls.ival(1,:) + 4*blk_ls.fs;
blk_ls.ival(2,:)= blk_ls.ival(2,:) - 1*blk_ls.fs;
mrk_r2= mrk_addBlockNumbers(mrk, blk_ls);
mrk_r2= mrk_selectEvents(mrk_r2, ~isnan(mrk_r2.block_no));

mrk_targets= mrk_mergeMarkers(mrk_r1, mrk_r2);

%% extract nontarget markers from pre-message intervals:
mrk_premsg= mrk_setClasses(mrk_targets, 1:numel(mrk_targets.pos), 'rest');
%mrk_premsg.pos= mrk_premsg.pos - bbci.setup_opts.disp_ival(2)/1000*mrk.fs;
mrk_premsg.pos= mrk_premsg.pos - mrk.fs;
%% extract nontarget markers during secondary task:
blk_cs= blk_segmentsFromMarkers(mrk_orig, ...
                                'start_marker','S 40','end_marker','S 41');
% correct for transition between tasks
blk_cs.ival(1,:)= blk_cs.ival(1,:) + 4*blk_cs.fs;
blk_cs.ival(2,:)= blk_cs.ival(2,:) - 1*blk_cs.fs;
mrk_cs= mrk_evenlyInBlocks(blk_cs, diff(bbci.setup_opts.disp_ival), 'offset_start',1000);
mrk_cs.y= ones(1, length(mrk_cs.pos));
mrk_cs.toe= mrk_cs.y;
mrk_cs.className= {'rest'};

mrk_nontargets= mrk_mergeMarkers(mrk_cs, mrk_premsg);
%mrk= mrk_mergeMarkers(mrk_targets, mrk_nontargets);

mrk_r1.className= {'t1'};
mrk_r2.className= {'t2'};
mrk_premsg.className= {'nt-premsg'};
mrk_cs.className= {'nt-ontask'};
mrk= mrk_mergeMarkers(mrk_r1, mrk_r2, mrk_premsg, mrk_cs);
