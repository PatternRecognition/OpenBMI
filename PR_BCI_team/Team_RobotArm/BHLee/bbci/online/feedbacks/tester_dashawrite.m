opt= struct('type','dashawrite');
speedup= 1;

opi= set_defaults(opt,'status','play','changed',1);
opt= feedback_dashawrite(gcf, opi, 0);
waitForSync;

ctrl= 0;
while 1,
C= interp(randn(1,500), 8);
for cc= 1:length(C),
  ws= waitForSync(1000/25/speedup);
  if ws>0,
    fprintf('lost: %.1f ms\n', ws);
  end
  ctrl= max(-1, min(1, ctrl + C(cc)/5));
  opt= feedback_dashawrite(gcf, opt, ctrl);
end
waitForSync(1000/25/speedup);
%opt= feedback_dashawrite(gcf, setfield(opt,'text_reset',1), ctrl);
end
