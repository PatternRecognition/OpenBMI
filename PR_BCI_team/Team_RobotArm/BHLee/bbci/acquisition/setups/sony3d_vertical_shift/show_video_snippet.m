function show_video_snippet(fname, duration, overlap, which_snippet, num_snippets)

global tcp_conn

len = duration / num_snippets + overlap;
start = 0:len-overlap:duration;
start = start(1:end-1);

pnet(tcp_conn, 'printf', 'vid %.3f %s', start(which_snippet), fname);
fprintf('waiting %.0f seconds before stopping video...\n',len);
pause(len);
pnet(tcp_conn, 'printf', 'vid 0');