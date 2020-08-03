%% feedback_opt, dtct_out, dscr_out, comb_out

feedback_opt.fs= cnt.fs/feedback_opt.step;
timeStep= 1000/feedback_opt.fs;

feedback_opt= animate_feedback(feedback_opt, 'init', dtct_out);
pp= test_begin;
waitForSync;
for ptr= 1:length(dtct_out),
  animate_feedback(feedback_opt, ptr, dtct_out, dscr_out, comb_out);
  waitForSync(timeStep);
  pp= pp + feedback_opt.step;
end
