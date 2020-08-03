function theta = augcog_threshold(Y, Yh, PERC)
% augcog_threshold
%
%   compute the theta threshold for the augcog traffic sign.
%
%   THETA = AUGCOG_THRESHOLD(Y, classifier_output, <perc>)
%
% THETA: 2-array of thresholds relative to the classifier output.
%
%   if perc is not set, it is determined by minimizing the training
%   hysteresis error (don't know if that is a good idea :)
%
% written Aug 6 2004 by Mikio Braun
%

th = sort(-Yh);

steps = abs(Yh(1) - Yh(end))/100;
th = sort([th - steps, th + steps]);

N = length(th);

%plot(1:N, th, 'o-');

Z = zeros(1, N);
for I = 1:N
  Z(I) = 1 - mean(Y ~= sign(Yh + th(I)));
  %Z(:,I) = sign(Yh + th(I))';
end

plot(th, Z)
MI = max(Z);

if nargin ~= 3,
  % find the optimal position by minimizing the hysteresis error
  PR = .5:.01:.99;
  
  HE = zeros(1, length(PR));
  
  for I = 1:length(PR),
    PERC = PR(I);
    
    RA = min(find(Z > MI*PERC));
    RB = max(find(Z > MI*PERC));

    theta(1) = th(RA);
    theta(2) = th(RB);
    
    Yht = sum(repmat(Yh, length(theta), 1) > repmat(theta', 1, size(Yh, 2)))-1;
    
    for J = 2:length(Yht)
      if Yht(J) == 0,
	Yht(J) = Yht(J-1);
      end
    end
    
    HE(I) = mean(Y ~= Yht);
  end

  plot(PR, HE);
  
  [dummy, I] = min(HE);
  
  PERC = PR(I);
end

RA = min(find(Z > MI*PERC));
RB = max(find(Z > MI*PERC));

theta(1) = th(RA);
theta(2) = th(RB);
