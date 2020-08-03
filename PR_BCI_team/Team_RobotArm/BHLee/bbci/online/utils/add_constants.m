function udp = add_constants(out, mrkOut,position,value);
% TODO: extended documentation by Schwaighase

udp = vertcat(out{:}, mrkOut{:});

for i = 1:length(position)
  udp = [udp(1:position(i)-1),value(i),udp(position(i):end)];
end

