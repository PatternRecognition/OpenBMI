clear opt
opt.log= 0;

opt.classes= {'left','down'};
opt.trials_per_run= 30;
opt.break_every= 10;
opt.break_show_score= 1;
opt.break_endswith_countdown= 1;
opt.duration_break= 2000;
opt.duration= 3000;
opt.duration_until_hit= 1000;
opt.duration_before_free= 1000;
opt.timeout_policy= 'hitiflateral';
opt.cursor_on= 1;
%opt.show_score= 1;
%opt.show_rejected= 1;
opt.damping_in_target= 'quadratic';
opt.punchline= 1;
opt.touch_terminates_trial= 0;
opt.countdown= 2000;

if 0, %% adaption
  opt.adapt_trials= 6;
  opt.adapt_duration_rotation= 3*1000;
end

speedup= 1;
%speedup= 2;

opi= set_defaults(opt, 'status','play', 'changed',1);
opt= feedback_cursor_arrow(gcf, opi, 0);
waitForSync;
for ii= 1:500;
  waitForSync(1000/25/speedup);
  ctrl= -1;
  opt= feedback_cursor_arrow(gcf, opt, ctrl);
end
for ii= 1:500;
  waitForSync(1000/25/speedup);
  ctrl= 1;
  opt= feedback_cursor_arrow(gcf, opt, ctrl);
end

while 1,
C= interp(randn(1,500), 8);
for cc= 1:length(C),
  ws= waitForSync(1000/25/speedup);
  if ws>0,
    fprintf('lost: %.1f ms\n', ws);
  end
  ctrl= max(-1, min(1, ctrl + C(cc)/5));
  opt= feedback_cursor_arrow(gcf, opt, ctrl);
end
end
