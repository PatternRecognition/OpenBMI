gsetup_file= 'hexawrite_lett_fast.gsetup';

file= which(gsetup_file);
fid= fopen(file);
F= fread(fid);
fclose(fid);

eval(char(F'));
fb_opt= setup.graphic_player.feedback_opt;
fb_opt.ctrl = {[29,28]};


fb_opt.order_sequence= '/home/neuro/data/BCI/temp/einzelne_buchstaben.txt';
fb_opt.log= 0;
