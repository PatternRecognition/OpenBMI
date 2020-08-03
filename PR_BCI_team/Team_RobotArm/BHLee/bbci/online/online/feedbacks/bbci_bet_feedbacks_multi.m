function udp = bbci_bet_feedbacks_multi(udp,bbci);

if isfield(bbci,'feedback_class_order') & ~isempty(bbci.feedback_class_order)
  udp = udp(bbci.feedback_class_order);
end

% if length(udp)>3
%   error('1d feedback requires a one-dimensional output');
% end

% if length(udp)==2
%   udp = diff(udp);
% end

% if length(udp)==3
%   [dum,ind] = max(udp);
%   switch ind
%    case 3
%     udp = 0;
%    case 1
%     udp = max(udp([2,3]))-udp(1);
%    case 2
%     udp = udp(2)-max(udp([1,3]));
%   end
% end


% [udp;0];