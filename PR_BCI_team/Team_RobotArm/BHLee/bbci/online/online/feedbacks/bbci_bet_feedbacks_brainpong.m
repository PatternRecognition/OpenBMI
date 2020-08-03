function udp = bbci_bet_feedbacks_brainpong(udp,bbci);

if isfield(bbci,'feedback_class_order') & ~isempty(bbci.feedback_class_order)
  udp = udp(bbci.feedback_class_order);
end
 
if length(udp)>3
  error('1d feedback requires a one-dimensional output');
end

if length(udp)==2
  udp = diff(udp);
end

if length(udp)==3
  udp = 0*(udp(3)==max(udp))+ diff(udp)*(udp(3)<max(udp));
end


udp = [bbci.player;udp];