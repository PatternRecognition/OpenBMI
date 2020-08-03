function udp = bbci_bet_feedbacks_1d_error(udp,bbci);

if isfield(bbci,'feedback_class_order') & ~isempty(bbci.feedback_class_order)
  udp(1:end-1) = udp(bbci.feedback_class_order);
end

if length(udp)>4
  error('1d feedback requires a one-dimensional output');
end

err = udp(end);

if length(udp)==2
udp = udp(1);
end

if length(udp)==3
  udp = udp(2)-udp(1);
end

if length(udp)==4
  [dum,ind] = max(udp(1:3));
  switch ind
   case 3
    udp = 0;
   case 1
    udp = max(udp([2,3]))-udp(1);
   case 2
    udp = udp(2)-max(udp([1,3]));
  end
end

udp = [udp;0;err];
