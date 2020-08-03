fb_opt = struct('position',[0 0 1280 1024]);
fb_opt.relational = 1;
fb_opt.type = 'feedback_speller_2d';
fb_opt.ctrl = {[28,29]};
fb_opt.target_width = 0.1;
fb_opt.damping = 20;
fb_opt.order_sequence = 'd:\matlab\eegStimulus\einzelne_buchstaben.txt';
fb_opt.tolerance_length = 0;
fb_opt.eval_deletes = false;
fb_opt.order_item_limit = inf;
fb_opt.order_sequence_limit = 600;
fb_opt.order_pause = 1000;
fb_opt.countdown = 2000;
fb_opt.buffertext_pos = [0 0.77];
fb_opt.language_model = 'german';


