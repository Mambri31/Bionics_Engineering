%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo programma grafica la ddp della stima varianza campionaria 
% e della sua approssimazione Gaussiana
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
sigma2=1;
N=10;
etax=sigma2;
varx=2*(sigma2^2)/N;
x=[0:0.001:3];
%
fxG=exp(-((x-etax).^2)./(2*varx))/sqrt(2*pi*varx);
k1=(N/(2*sigma2))^((N-1)/2);
k2=prod(1:2:N-2)*sqrt(pi)/(2^((N-2)/2));
fxCQ=(k1/k2)*(x.^((N-1)/2-1)).*exp(-N/(2*sigma2).*x);
%
figure (1);
plot(x,fxCQ,'b','Linewidth', 2);
hold on; grid on;
plot(x,fxG,'r','Linewidth', 2);
legend('Sample Variance pdf', 'Gaussian approximation');
title('Probability Density Function of the Sample Variance');
xlabel('x');ylabel('f_{X}(x)')
axis([0 3 0 2])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%