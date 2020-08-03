opt= struct('type','hexawrite_wenke');
opt.labelset= ['ABCDE'; 'FGHIJ'; 'KLMNO'; 'PQRST'; 'UVWXY'; 'Z_<.?'];
opt.show_control= 1;
opt.order_sequence= {'DER','DES','','DER','APFEL','FAELLT','HEUTE','BESONDERS','WEIT','VOM','STAMM',''};
opt.begin_text= 1;
%opt.order_item_limit= 1;
speedup= 3;
opt.position= [50 10 1280 1024];

opt= feedback_hexawrite_wenke(gcf, opt, 0);
waitForSync;
opt.status= 'play';
opt.changed= 1;
opt= feedback_hexawrite_wenke(gcf, opt, 0);
waitForSync;
ctrl= 1;
for ii= 1:1000,
  waitForSync(1000/25/speedup);
  opt= feedback_hexawrite_wenke(gcf, opt, ctrl);
end

while 1,
C= interp(randn(1,500), 8);
for cc= 1:length(C),
  ws= waitForSync(1000/25/speedup);
  if ws>0,
    fprintf('lost: %.1f ms\n', ws);
  end
  ctrl= max(-1, min(1, ctrl + C(cc)/5)) + 0.05;
  opt= feedback_hexawrite_wenke(gcf, opt, ctrl);
end
waitForSync(1000/25/speedup);
%opt= feedback_hexawrite_wenke(gcf, setfield(opt,'text_reset',1), ctrl);
end
