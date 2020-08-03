function show_time(i,i_tot)

if mod(i,i_tot/10)==0
disp(sprintf('%i of %i done',i,i_tot))
end