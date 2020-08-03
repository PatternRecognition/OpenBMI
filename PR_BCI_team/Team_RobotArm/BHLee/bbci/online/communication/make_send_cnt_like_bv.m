try,
  send_cnt_like_bv('close');
end
if isunix
    mex -g send_cnt_like_bv.c
else
    mex send_cnt_like_bv.c ws2_32.lib
end