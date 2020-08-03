clear opt
opt.free_trials= 0;
opt.log= 0;

opt.trials_per_run= 30;
opt.response_at= 'none';
opt.break_every= 3;
opt.break_show_score= 0;
opt.break_endswith_countdown= 1;
opt.duration_break= 2000;
opt.duration= 2000;
opt.duration_jitter= 0;
opt.duration_before_free= 1000;
opt.cursor_on= 1;
opt.show_bit= 0;

opt.touch_terminates_trial= 0;

if 0, %% adaption
  opt.adapt_trials= 6;
  opt.adapt_duration_rotation= 3*1000;
end

speedup= 1;
%speedup= 2;

opi= set_defaults(opt, 'status','play', 'changed',1);
opt= feedback_cursor_1d_pro(gcf, opi, 0);
waitForSync;
for ii= 1:500;
  waitForSync(1000/25/speedup);
  ctrl= -1;
  opt= feedback_cursor_1d_pro(gcf, opt, ctrl);
end
for ii= 1:500;
  waitForSync(1000/25/speedup);
  ctrl= 1;
  opt= feedback_cursor_1d_pro(gcf, opt, ctrl);
end

while 1,
C= interp(randn(1,500), 8);
for cc= 1:length(C),
  ws= waitForSync(1000/25/speedup);
  if ws>0,
    fprintf('lost: %.1f ms\n', ws);
  end
  ctrl= max(-1, min(1, ctrl + C(cc)/5));
  opt= feedback_cursor_1d_pro(gcf, opt, ctrl);
end
end
