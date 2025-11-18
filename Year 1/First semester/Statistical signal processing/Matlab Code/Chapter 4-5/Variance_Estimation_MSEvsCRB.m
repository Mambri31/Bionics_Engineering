%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo programma grafica l'MSE dello stimatore ML della varianza e della
% sua versione scalata non polarizzata e li confronta con il CRB
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

eta=2;
varx=1;
N=[2:2:100]';
MSE_ML=(2*N-1)*(varx^2)./(N.^2);
MSE_NP=2*(varx^2)./(N-1);
CRB=2*(varx^2)./N;

figure (1);
loglog(N,MSE_ML,'b','Linewidth', 2);grid on;
hold on
loglog(N,MSE_NP,'g','Linewidth', 2);
loglog(N,CRB,'r-.','Linewidth', 2);
legend('MSE ML Estimator', 'MSE Unbiased Estimator', 'CRB');
title('MSE vs CRB');
xlabel('Data Size (N)');ylabel('MSE, CRB')
axis([2 100 0.01 2])

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