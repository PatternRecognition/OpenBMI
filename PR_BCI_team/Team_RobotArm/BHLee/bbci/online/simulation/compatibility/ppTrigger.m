function ppTrigger(x)

v= clock;
time_str= sprintf('%02d:%02d:%06.3f', v(4), v(5), v(6));
fprintf('trig #%03d at %s\n', x, time_str);
