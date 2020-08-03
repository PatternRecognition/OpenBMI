gsetup_file= 'speller_word_fast.gsetup';

file= which(gsetup_file);
fid= fopen(file);
F= fread(fid);
fclose(fid);

eval(char(F'));
fb_opt= setup.graphic_player.feedback_opt;
fb_opt.damping = 20;
fb_opt.relational = 1;
