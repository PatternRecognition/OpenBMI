opt= struct('type','speller_2d');
opt.countdown= 3000;
opt.order_sequence= {'DER','DES','','DER','APFEL','FAELLT','HEUTE','BESONDERS','WEIT','VOM','STAMM',''};
opt.begin_text= 1;
%opt.order_item_limit= 1;
speedup= 1;

opt= feedback_speller_2d_wenke(gcf, opt, 0);
waitForSync;
opt.status= 'play';
opt.changed= 1;
opt= feedback_speller_2d_wenke(gcf, opt, 0);
waitForSync;

while 1,
C= interp(randn(1,500), 8);
for cc= 1:length(C),
  waitForSync(1000/25/speedup);
  ctrl= max(-1, min(1, ctrl + C(cc)/5)) - 0.05;
  opt= feedback_speller_2d_wenke(gcf, opt, ctrl);
end
waitForSync(1000/25/speedup);
end
