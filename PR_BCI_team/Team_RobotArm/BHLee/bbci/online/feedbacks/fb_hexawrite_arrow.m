function arrow= fb_hexawrite_arrow(angle, len, opt)

arrow_length= opt.arrow_minlength + len*(1-opt.arrow_minlength);
w= [sin(angle); cos(angle)]*opt.hexradius;
wn= [-w(2); w(1)]/sqrt(w'*w)*opt.hexradius;
bp= -w*opt.arrow_backlength;
pp= w*arrow_length;
fp= w*(arrow_length-opt.arrow_headlength);

arrow= [bp+wn*opt.arrow_width, ...
	fp+wn*opt.arrow_width, fp+wn*opt.arrow_headwidth, pp, ...
	fp-wn*opt.arrow_headwidth, fp-wn*opt.arrow_width, ...
	bp-wn*opt.arrow_width];



  
     