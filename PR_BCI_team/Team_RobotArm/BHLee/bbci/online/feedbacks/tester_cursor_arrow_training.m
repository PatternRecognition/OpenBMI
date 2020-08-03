clear opt
opt.log= 0;

opt.trigger_classifier_list= {{1,2},{1,3},{3,2}};
opt.trials_per_run= 30;
opt.break_every= 15;
opt.break_endswith_countdown= 1;
opt.duration_break= 2000;
opt.duration= 3000;
opt.duration_until_hit= 3000;
opt.duration_before_free= 1000;
opt.cursor_on= 1;
opt.countdown= 2000;
opt.fallback= 0.3;

speedup= 1;
%speedup= 2;

opi= set_defaults(opt, 'status','play', 'changed',1, 'reset',1);
opt= feedback_cursor_arrow_training(gcf, opi, 0);
waitForSync;

ctrl1= 0;
ctrl2= 0;
ctrl3= 0;
while 1,
C1= interp(randn(1,500), 8);
C2= interp(randn(1,500), 8);
C3= interp(randn(1,500), 8);
for cc= 1:length(C1),
  ws= waitForSync(1000/25/speedup);
  if ws>0,
    fprintf('lost: %.1f ms\n', ws);
  end
  ctrl1= max(-1, min(1, ctrl1 + C1(cc)/5));
  ctrl2= max(-1, min(1, ctrl1 + C1(cc)/5));
  ctrl3= max(-1, min(1, ctrl1 + C1(cc)/5));
  opt= feedback_cursor_arrow_training(gcf, opt, ctrl1, ctrl2, ctrl3);
end
end
