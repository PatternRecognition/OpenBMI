function udp = bbci_bet_feedbacks_2d(udp,bbci);

if isfield(bbci,'feedback_class_order') & ~isempty(bbci.feedback_class_order)
  udp = udp(bbci.feedback_class_order);
end

if length(udp)>4 | length(udp)<2
  error('2d feedback requires a two-four-dimensional output');
end

if length(udp)==4
  if isfield(bbci,'one_direction') & bbci.one_direction
    [dum,ind] = sort(udp);
    udp = udp(ind(end))-udp(ind(end-1));
    if ind(end)<=2
      udp(2) = [udp,0];
    else
      udp(1) = [0,udp];
    end
  else
    udp = [udp(2)-udp(1),udp(4)-udp(3)];
  end
  return;
end

if length(udp)==2
  if isfield(bbci,'one_direction') & bbci.one_direction
    if abs(udp(1))>abs(udp(2))
      udp = [sign(udp(1))*(abs(udp(1))-abs(udp(2))),0];
    else
      udp = [0,sign(udp(2))*(abs(udp(2))-abs(udp(1)))];
    end
  else
    % nothing!!!
  end
  return;
end


if length(udp)==3
  if isfield(bbci,'star') & ~bbci.star
    udp = udp([1,2])-udp(3);
    return
  end
  if isfield(bbci,'one_direction') & bbci.one_direction
    [dum,ind] = sort(udp);
    u = udp(ind(end))-udp(ind(end-1));
    udp = zeros(1,3);
    udp(ind(end)) = u;
  else
    udp = udp-min(udp);
  end
  udp = [-1,1,0;cot(pi/3),cot(pi/3),-1]*udp;
  return;
end



      
      


