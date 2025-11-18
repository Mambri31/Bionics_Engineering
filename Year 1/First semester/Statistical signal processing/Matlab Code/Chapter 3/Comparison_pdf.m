%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This matlab script plots 3 different distirbutinos: 
% Gaussian, Laplace and uniform
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

x=[-6:0.0001:6]';
g=exp(-x.^2./2)/sqrt(2*pi);
l=exp(-sqrt(2)*abs(x))./sqrt(2);
u=rectpuls(x./(2*sqrt(3)))./(2*sqrt(3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure (1);
grid on
plot(x,g,'LineWidth',2);grid on;
hold on
plot(x,l,'g','LineWidth',2);grid on;
hold on
plot(x,u,'r','LineWidth',2);
legend('Gaussian pdf','Laplace pdf','Uniform pdf');
title('Comparison of the three zero mean and unit variance distributions');xlabel('x');ylabel('f_{X} (x), f_{Y} (x), f_{Z} (x)')
axis([3 6 0 0.01])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

x=[-6:0.001:6]';
g=exp(-x.^2./2)/sqrt(2*pi);
l=exp(-sqrt(2)*abs(x))./sqrt(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure (2);
semilogy(x,g,'LineWidth',2);grid on;
hold on
semilogy(x,l,'g','LineWidth',2);grid on;
grid on
legend('Gaussian pdf','Laplace pdf');
title('Comparison of the two zero mean and unit variance distributions');xlabel('x');ylabel('f_{X} (x), f_{Y} (x)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%
% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
