%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola la stima LMMSE del vettore di segnale S
% dato il vettore degli osservati X=S+W
% S[n] è modellato come un processo AR(1), W[n] è AWGN
% Filtro di Wiener di tipo Smoothing
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N=300;                  % Numero di campioni osservati
varS=1;                 % Varianza del segnale
rho=0.2;                % Signal correlation coefficient - AR(1) process
GammadB=-10;             % Rapporto varianza segnale - varianza rumore in decibel [dB] 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gamma=10^(GammadB/10);        % Gamma in scala lineare 
varW=varS/Gamma;              % Potenza di rumore 
m=[0:N-1]';                   % Vettore del tempo-discreto
Cw=varW*eye(N);               % Noise covariance matrix
Cs=varS*toeplitz((rho.^m)');  % Signal covariance matrix
A=Cs*inv(Cs+Cw);              % Wiener Smoothing Matrix
L=chol(Cs)';                  % L*L'=Cs Decomposizione di Choleski
W=sqrt(varW)*randn(N,1);      % Generazione vettore di rumore AWGN
S=L*randn(N,1);               % Generazione vettore di segnale Gaussiano AR(1)
X=S+W;                        % Vettore dei dati osservati
Shat=A*X;                     % Stima LMMSE di S dato X
Cerr=(eye(N)-A)*Cs;           % Matrice di covarianza dell'errore di stima

figure (1);
plot(m,X,'b', 'Linewidth', 2);
grid on;
hold on;
plot(m,S,'r-', 'Linewidth', 2);
legend('Segnale osservato X[n]','Segnale S[n]');
title('Segnali X[n] ed S[n] vs N');xlabel('Sample size (N)');ylabel('Segnali X[n] ed S[n]')

figure (2);
plot(m,Shat,'g', 'Linewidth', 2);
grid on;
hold on;
plot(m,S,'r-', 'Linewidth', 2);
legend('Stima LMMSE di S[n]','Segnale S[n]');
title('S[n] e sua stima LMMSE vs N');xlabel('Sample size (N)');ylabel('S[n] e stima LMMSE di S[n]')

figure (3);
plot(m,diag(Cerr),'g', 'Linewidth', 2);
grid on;
hold on;
legend('MMSE of S[n] estimate');
title('MMSE vs n');xlabel('Discrete-Time (n)');ylabel('MMSE[n]')
axis([1 300 0 1])

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

 