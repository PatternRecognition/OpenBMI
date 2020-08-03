function udp = bbci_bet_feedbacks_2dstraight(udp,bbci);

if isfield(bbci,'feedback_class_order') & ~isempty(bbci.feedback_class_order)
  udp = udp(bbci.feedback_class_order);
end

if length(udp)~=3,
  keyboard
  error('expecting 3-dim input');
end

dp(1)= udp(1) - mean(udp([2 3]));
dp(2)= udp(2) - mean(udp([1 3]));
dp(3)= udp(3) - mean(udp([1 2]));
udp= reshape(dp, size(udp));
