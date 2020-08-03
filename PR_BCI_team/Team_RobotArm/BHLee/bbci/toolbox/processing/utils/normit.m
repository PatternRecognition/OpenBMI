function [A,D]=normit(A);
 
%normalize columns to 1
%by appropriate scaling  
   D=diag( 1./sqrt(diag(A'*A) ) );
   
   A=A*D;
   
   