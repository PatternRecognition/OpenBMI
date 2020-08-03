function r = lin_sample(n, l_border, m_border, r_border)
% LIN_SAMPLE Summary of this function goes here
%    Detailed explanation goes here

interval1 = (m_border-l_border);
interval2 = (r_border-m_border);

r = (r_border-m_border) * rand(1,n) + m_border;
f = (interval1/2)/((interval1/2)+ interval2);

 ixDiag = find(rand(1,n) < f);
 r(ixDiag) = (m_border-l_border) * sqrt(rand(1,length(ixDiag))) +l_border;
