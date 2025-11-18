%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Questo programma grafica le ddp dei due stimatori "primo campione" e "media campionaria" del valor medio 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;

eta=2;
var=1;
N=20;
varN=var/N;

x=[eta-5*sqrt(var):0.001:eta+5*sqrt(var)];
fx1=exp(-((x-eta).^2)./(2*var))/sqrt(2*pi*var);
fx2=exp(-((x-eta).^2)./(2*varN))/sqrt(2*pi*varN);

% Figura rappresentante le ddp dei due stimatori

figure (1);
plot(x,fx1,'b','Linewidth', 2);grid on;
hold on
plot(x,fx2,'g','Linewidth', 2);
legend('pdf of the estimator 1 sample', 'pdf estimator Sample Mean');
title('Distributions of the estimators');
xlabel('\eta');ylabel('f_{\eta}(\eta)')
axis([-3 7 0 2.5])

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

 
