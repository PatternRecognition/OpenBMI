opt= struct('type','hexawrite', 'log',0);
opt.labelset= ['ABCDE'; 'FGHIJ'; 'KLMNO'; 'PQRST'; 'UVWXY'; 'Z_<.?'];
opt.show_control= 1;
speedup= 1;

opi= set_defaults(opt,'status','play','changed',1);
opt= feedback_hexawrite(gcf, opi, 0);
waitForSync;
for ii= 1:85;
  waitForSync(1000/25/speedup);
  ctrl= -0.5;
  opt= feedback_hexawrite(gcf, opt, ctrl);
end
for ii= 1:35;
  waitForSync(1000/25/speedup);
  ctrl= 1;
  opt= feedback_hexawrite(gcf, opt, ctrl);
end
for ii= 1:60;
  waitForSync(1000/25/speedup);
  ctrl= -0.5;
  opt= feedback_hexawrite(gcf, opt, ctrl);
end
for ii= 1:25;
  waitForSync(1000/25/speedup);
  ctrl= 1;
  opt= feedback_hexawrite(gcf, opt, ctrl);
end

while 1,
C= interp(randn(1,500), 8);
for cc= 1:length(C),
  ws= waitForSync(1000/25/speedup);
  if ws>0,
    fprintf('lost: %.1f ms\n', ws);
  end
  ctrl= max(-1, min(1, ctrl + C(cc)/5)) + 0.05;
  opt= feedback_hexawrite(gcf, opt, ctrl);
end
waitForSync(1000/25/speedup);
%opt= feedback_hexawrite(gcf, setfield(opt,'text_reset',1), ctrl);
end
