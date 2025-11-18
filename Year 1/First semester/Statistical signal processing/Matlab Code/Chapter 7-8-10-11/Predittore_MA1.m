%
% Questo programma genera un processo MA(1) ed implementa il Predittore ad
% 1 passo
%
clear all;
close all;
%
% MA(1) process
%
b1=-0.98;
varX=1;
varW=varX/(1+b1^2)
N=1000;
index=[1:N];
w=sqrt(varW)*randn(1,N);

x=filter([1 b1]',1,w);

xhat=filter([0 b1]',[1 b1]',x);
%
% Figura rappresentante una possibile realizzazione
%
figure(1)
plot(0:N-1,x,'g','Linewidth', 2);grid on;
hold on
plot(0:N-1,xhat,'r','Linewidth', 2);grid on;
axis([N-100 N -3 3]);
legend('X[n]','Predizione ad 1 passo di X[n]');
title('Processo MA(1) X[n] e Predizione ad 1 passo di X[n]')
xlabel('Discrete-Time [n]')
ylabel('X[n], Predizione ad 1 passo di X[n]')
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%