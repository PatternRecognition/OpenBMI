clear opt
opt.trials_per_run= 12;
opt.cursor_on= 1;
opt.response_at= 'cursor';
opt.break_every= 3;
opt.duration_break= 2000;
opt.free_trials= 0;
opt.log= 0;
opt.duration_jitter= 1200;
speedup= 2;

opi= set_defaults(opt, 'status','play', 'changed',1);
opt= feedback_cursor_1d_fixed_dur(gcf, opi, 0);
waitForSync;
for ii= 1:500;
  waitForSync(1000/25/speedup);
  ctrl= -1;
  opt= feedback_cursor_1d_fixed_dur(gcf, opt, ctrl);
end
for ii= 1:500;
  waitForSync(1000/25/speedup);
  ctrl= 1;
  opt= feedback_cursor_1d_fixed_dur(gcf, opt, ctrl);
end

while 1,
C= interp(randn(1,500), 8);
for cc= 1:length(C),
  ws= waitForSync(1000/25/speedup);
  if ws>0,
    fprintf('lost: %.1f ms\n', ws);
  end
  ctrl= max(-1, min(1, ctrl + C(cc)/5));
  opt= feedback_cursor_1d_fixed_dur(gcf, opt, ctrl);
end
end
