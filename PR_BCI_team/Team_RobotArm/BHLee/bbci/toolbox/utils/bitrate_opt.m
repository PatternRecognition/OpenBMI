function val = bitrate_opt(p,uber)

p = [p;1-p];

val = uber*p;

val = uber./repmat(val,[1,2]);

val(val==0)=1;
val = log2(val);

val = val.*uber;

val = val*p;
val = -sum(val);

