function t_crit= calcTcrit(alpha, nu)
%calcTcrit - calculate threshold values for Student's t-test
%
%Usage:
%  T_CRIT= calcTcrit(ALPHA, NU)
%
%Input:
%  ALPHA:  level to significance. Can also be a vector to calculate
%          several thresholds at the same time.
%  NU:     degrees of freedom
%
%Output:
%  T_CRIT: Threshold value(s)
%
%See also proc_t_scale, proc_alpha_steps

if length(alpha)>1,
  for ii= 1:length(alpha),
    t_crit(ii)= calcTcrit(alpha(ii), nu);
  end
  return;
end

xi= linspace(0, 1, 1000);
be= betainc(xi, nu/2, 1/2);
cr= max(find(be<2*alpha));

xi= linspace(xi(cr), xi(cr+1), 1000);
be= betainc(xi, nu/2, 1/2);
cr= max(find(be<2*alpha));

xi= linspace(xi(cr), xi(cr+1), 1000);
be= betainc(xi, nu/2, 1/2);
cr= max(find(be<2*alpha));
xi_crit= xi(cr);

t_crit= sqrt(nu/xi_crit-nu);
