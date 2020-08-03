function cc= combinerDal(fb_opt, ptr, dscr_out, dtct_out, comb_out)
%comb_out= combinerDal(fb_opt, ptr, dscr_out, dtct_out, comb_out)

persistent detect

if ptr==1,
  detect= 0;
end

cc= 0;
p0= max(1, ptr-fb_opt.integrate+1);
mn= mean(dtct_out(p0:ptr));
lr= mean(dscr_out(p0:ptr));
if ~detect,
  if mn>0 & mn+abs(lr)>1 & abs(lr)>0.25,
    detect= 1;
    cc= sign(lr);
  end
else
  if sign(lr)~=comb_out(ptr-1) | ...
        (mn<0.25 & abs(lr)<0.5),
    detect= 0;
  else
    cc= comb_out(ptr-1);
  end
end
