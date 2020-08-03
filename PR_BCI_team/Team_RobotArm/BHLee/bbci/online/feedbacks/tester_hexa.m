opt= struct('type','hexa');
opt.position= [0 266 1280 753];
speedup= 1;

opi= set_defaults(opt,'status','play','changed',1);
opt= feedback_hexa(gcf, opi, 0);
ctrl= 0;
waitForSync;
while 1,
C= interp(randn(1,40), 8);
opt.cebit_layout= 1;
opt.changed= 1;
for cc= 1:length(C),
  ws= waitForSync(1000/25/speedup);
  if ws>0,
    fprintf('lost: %.1f ms\n', ws);
  end
  ctrl= max(-1, min(1, ctrl + C(cc)/5)) + 0.045;
  opt= feedback_hexa(gcf, opt, ctrl);
end
opt.cebit_layout= 0;
opt.changed= 1;
for cc= 1:length(C),
  ws= waitForSync(1000/25/speedup);
  if ws>0,
    fprintf('lost: %.1f ms\n', ws);
  end
  ctrl= max(-1, min(1, ctrl + C(cc)/5)) + 0.045;
  opt= feedback_hexa(gcf, opt, ctrl);
end
waitForSync(1000/25/speedup);
%opt= feedback_hexa(gcf, setfield(opt,'text_reset',1), ctrl);
end
