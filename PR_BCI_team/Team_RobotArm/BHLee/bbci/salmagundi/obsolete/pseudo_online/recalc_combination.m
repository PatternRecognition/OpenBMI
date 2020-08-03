comb_out= zeros(test_len, 1);

ptr= 1;
for pp= test_begin:feedback_opt.step:test_end,
  comb_out(ptr)= feval(feedback_opt.combiner_fcn, feedback_opt, ...
                       ptr, dtct_out, dscr_out, comb_out);
  ptr= ptr+1;
end

