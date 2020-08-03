function t_crit= calcTcrit(alpha, nu)
%t= calcTcrit(alpha, nu)

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
