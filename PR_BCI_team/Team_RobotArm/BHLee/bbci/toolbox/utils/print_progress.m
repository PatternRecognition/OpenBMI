function print_progress(n, N)
%print_progress(n, N)
%
% print work progress of loops (tic, for n=1:N)

%% idea by stefan harmeling

cent= round(100*n/N);
togo= toc/n*(N-n);
fprintf('\r%3d%%, %ds to go    ', cent, floor(togo));
if n==N,
  fprintf('\r%40s\r','');
end
