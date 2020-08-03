function udp = bbci_bet_feedbacks_LibetGame(udp, bbci);

persistent old_timestamp

classifier = udp(1);
marker = udp(2);
timestamp = udp(3);

if timestamp == bbci.minDataLength,
  old_timestamp= -inf;
end

time_passed= (timestamp-old_timestamp)/bbci.fs*1000;
toe= adminMarker('query', [-time_passed 0]);
keypressed= ismember(-1, toe);

control= 2*(classifier<0) + keypressed;
udp = int16(control);
